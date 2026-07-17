use crate::error::LoadError;

const BINARY_SHORTLIST_MAGIC: u64 = 0xF11A_48D5_0134_17F5;

pub(crate) struct Shortlist {
    /// Number of globally frequent target tokens included at runtime.
    first_num: usize,
    offsets: Vec<usize>,
    target_ids: Vec<u32>,
}

impl Shortlist {
    /// Parse a Bergamot binary lexical shortlist.
    pub(crate) fn load(bytes: Vec<u8>) -> Result<Self, LoadError> {
        if bytes.len() < 48 {
            return Err(LoadError::InvalidShortlist(format!(
                "invalid byte length {}",
                bytes.len()
            )));
        }

        let magic = u64_at(&bytes, 0)?;
        let expected_checksum = u64_at(&bytes, 8)?;
        if magic != BINARY_SHORTLIST_MAGIC {
            return Err(LoadError::InvalidShortlist(format!("bad magic {magic:#x}")));
        }

        let first_num = usize_value(u64_at(&bytes, 16)?)?;
        // This generation parameter is baked into the candidate table. Parse
        // it to validate the header representation, but runtime lookup does
        // not need to retain it or truncate candidates again.
        let _best_num = usize_value(u64_at(&bytes, 24)?)?;
        let offset_count = usize_value(u64_at(&bytes, 32)?)?;
        let target_count = usize_value(u64_at(&bytes, 40)?)?;
        let expected_len = 48usize
            .checked_add(offset_count.checked_mul(8).ok_or_else(|| {
                LoadError::InvalidShortlist("offset table length overflow".into())
            })?)
            .and_then(|n| n.checked_add(target_count.checked_mul(4)?))
            .ok_or_else(|| LoadError::InvalidShortlist("shortlist length overflow".into()))?;
        if bytes.len() != expected_len {
            return Err(LoadError::InvalidShortlist(format!(
                "header expects {expected_len} bytes, got {}",
                bytes.len()
            )));
        }

        let actual_checksum = marian_hash_u64(&bytes[16..]);
        if actual_checksum != expected_checksum {
            return Err(LoadError::InvalidShortlist(format!(
                "checksum mismatch: expected {expected_checksum:#x}, got {actual_checksum:#x}"
            )));
        }

        let mut offsets = Vec::with_capacity(offset_count);
        let mut cursor = 48;
        for _ in 0..offset_count {
            offsets.push(usize_value(u64_at(&bytes, cursor)?)?);
            cursor += 8;
        }
        if offsets.is_empty() || offsets.last().copied() != Some(target_count) {
            return Err(LoadError::InvalidShortlist(
                "last offset does not equal target ID count".into(),
            ));
        }
        if offsets
            .windows(2)
            .any(|pair| pair[0] > pair[1] || pair[1] > target_count)
        {
            return Err(LoadError::InvalidShortlist(
                "offset table is not monotonic".into(),
            ));
        }

        let mut target_ids = Vec::with_capacity(target_count);
        for _ in 0..target_count {
            target_ids.push(u32_at(&bytes, cursor)?);
            cursor += 4;
        }

        Ok(Self {
            first_num,
            offsets,
            target_ids,
        })
    }

    #[must_use]
    pub(crate) fn source_vocab_size(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    #[must_use]
    fn candidates(&self, source_id: usize) -> Option<&[u32]> {
        let start = *self.offsets.get(source_id)?;
        let end = *self.offsets.get(source_id + 1)?;
        self.target_ids.get(start..end)
    }

    pub(crate) fn generate_shared(
        &self,
        source_ids: &[u32],
        target_vocab_size: usize,
        shared_vocab: bool,
    ) -> Vec<u32> {
        self.generate_impl(source_ids, target_vocab_size, shared_vocab)
    }

    fn generate_impl(
        &self,
        source_ids: &[u32],
        target_vocab_size: usize,
        shared_vocab: bool,
    ) -> Vec<u32> {
        let mut selected = vec![false; target_vocab_size];
        let mut seen_source = vec![false; self.source_vocab_size()];
        for selected in selected
            .iter_mut()
            .take(self.first_num.min(target_vocab_size))
        {
            *selected = true;
        }
        for &source_id in source_ids {
            // Marian includes the source token itself when both sides use the
            // same SentencePiece model. This preserves names, numbers and
            // other copyable tokens even if the lexical table omits them.
            if shared_vocab && let Some(slot) = selected.get_mut(source_id as usize) {
                *slot = true;
            }
            let source_index = source_id as usize;
            if seen_source.get(source_index).copied().unwrap_or(false) {
                continue;
            }
            if let Some(seen) = seen_source.get_mut(source_index) {
                *seen = true;
            }
            if let Some(candidates) = self.candidates(source_index) {
                for &target_id in candidates {
                    if let Some(slot) = selected.get_mut(target_id as usize) {
                        *slot = true;
                    }
                }
            }
        }

        let mut count = selected.iter().filter(|&&value| value).count();
        for slot in selected.iter_mut().skip(self.first_num) {
            if count % 8 == 0 {
                break;
            }
            if !*slot {
                *slot = true;
                count += 1;
            }
        }

        selected
            .into_iter()
            .enumerate()
            .filter_map(|(id, selected)| selected.then_some(id as u32))
            .collect()
    }

    pub(crate) fn validate_target_vocab(&self, target_vocab_size: usize) -> Result<(), LoadError> {
        if let Some(&id) = self
            .target_ids
            .iter()
            .find(|&&id| id as usize >= target_vocab_size)
        {
            return Err(LoadError::InvalidShortlist(format!(
                "target ID {id} exceeds target vocabulary {target_vocab_size}"
            )));
        }
        Ok(())
    }
}

fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, LoadError> {
    let chunk = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| LoadError::InvalidShortlist("unexpected end of shortlist header".into()))?;
    let bytes = <[u8; 8]>::try_from(chunk)
        .map_err(|_| LoadError::InvalidShortlist("invalid 64-bit field".into()))?;
    Ok(u64::from_le_bytes(bytes))
}

fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, LoadError> {
    let chunk = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| LoadError::InvalidShortlist("unexpected end of target IDs".into()))?;
    let bytes = <[u8; 4]>::try_from(chunk)
        .map_err(|_| LoadError::InvalidShortlist("invalid target ID".into()))?;
    Ok(u32::from_le_bytes(bytes))
}

fn usize_value(value: u64) -> Result<usize, LoadError> {
    usize::try_from(value)
        .map_err(|_| LoadError::InvalidShortlist("integer does not fit usize".into()))
}

fn marian_hash_u64(bytes: &[u8]) -> u64 {
    // Marian computes hashMem<uint64_t> with an element count obtained using
    // integer division. Consequently, a final four-byte WordIndex is not part
    // of the checksum when the target list contains an odd number of entries.
    let mut seed = 0u64;
    for chunk in bytes.chunks_exact(8) {
        let mut value_bytes = [0; 8];
        value_bytes.copy_from_slice(chunk);
        let value = u64::from_le_bytes(value_bytes);
        seed ^= value
            .wrapping_add(0x9e37_79b9)
            .wrapping_add(seed << 6)
            .wrapping_add(seed >> 2);
    }
    seed
}

#[cfg(test)]
mod tests {
    use super::{BINARY_SHORTLIST_MAGIC, Shortlist, marian_hash_u64};

    fn binary(first_num: u64, best_num: u64, offsets: &[u64], ids: &[u32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&BINARY_SHORTLIST_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&first_num.to_le_bytes());
        bytes.extend_from_slice(&best_num.to_le_bytes());
        bytes.extend_from_slice(&(offsets.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&(ids.len() as u64).to_le_bytes());
        for offset in offsets {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        for id in ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        let checksum = marian_hash_u64(&bytes[16..]);
        bytes[8..16].copy_from_slice(&checksum.to_le_bytes());
        bytes
    }

    fn load(bytes: Vec<u8>) -> Result<Shortlist, crate::error::LoadError> {
        Shortlist::load(bytes)
    }

    #[test]
    fn loads_valid_shortlist() {
        let shortlist = load(binary(2, 3, &[0, 2, 3], &[4, 5, 6])).unwrap();
        assert_eq!(shortlist.first_num, 2);
        assert_eq!(shortlist.source_vocab_size(), 2);
        assert_eq!(shortlist.candidates(0), Some([4, 5].as_slice()));
        assert_eq!(shortlist.candidates(1), Some([6].as_slice()));
    }

    #[test]
    fn rejects_invalid_magic() {
        let mut bytes = binary(1, 1, &[0, 0], &[]);
        bytes[0] ^= 1;
        assert!(load(bytes).is_err());
    }

    #[test]
    fn rejects_checksum_mismatch() {
        let mut bytes = binary(1, 1, &[0, 1], &[2]);
        bytes[16] ^= 1;
        assert!(load(bytes).is_err());
    }

    #[test]
    fn accepts_odd_target_count_checksum() {
        let shortlist = load(binary(1, 1, &[0, 1], &[7])).unwrap();
        assert_eq!(shortlist.candidates(0), Some([7].as_slice()));
    }

    #[test]
    fn rejects_invalid_offsets() {
        assert!(load(binary(1, 1, &[0, 2, 1], &[4])).is_err());
        assert!(load(binary(1, 1, &[0, 0], &[4])).is_err());
    }

    #[test]
    fn rejects_target_ids_outside_vocabulary() {
        let shortlist = load(binary(1, 1, &[0, 1], &[8])).unwrap();
        assert!(shortlist.validate_target_vocab(8).is_err());
        shortlist.validate_target_vocab(9).unwrap();
    }

    #[test]
    fn generates_frequent_and_lexical_candidates() {
        let shortlist = Shortlist {
            first_num: 2,
            offsets: vec![0, 1, 2],
            target_ids: vec![10, 12],
        };
        let generated = shortlist.generate_shared(&[0, 1], 16, false);
        assert!(generated.starts_with(&[0, 1]));
        assert!(generated.contains(&10));
        assert!(generated.contains(&12));
        assert_eq!(generated.len() % 8, 0);
    }

    #[test]
    fn shared_generation_copies_deduplicates_and_sorts() {
        let shortlist = Shortlist {
            first_num: 1,
            offsets: vec![0; 17],
            target_ids: Vec::new(),
        };
        assert_eq!(
            shortlist.generate_shared(&[9, 9], 16, true),
            [0, 1, 2, 3, 4, 5, 6, 9]
        );
    }
}
