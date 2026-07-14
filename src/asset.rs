use std::io::{Cursor, Read};

use flate2::read::GzDecoder;

use crate::error::LoadError;

/// Compression applied to an in-memory model asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None,
    Gzip,
}

/// An owned asset passed across the IO-free core API boundary.
#[derive(Debug)]
pub struct Asset {
    pub bytes: Vec<u8>,
    pub compression: Compression,
}

impl Asset {
    #[must_use]
    pub fn raw(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            compression: Compression::None,
        }
    }

    #[must_use]
    pub fn gzip(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            compression: Compression::Gzip,
        }
    }

    pub(crate) fn into_reader(self) -> Box<dyn Read + Send> {
        match self.compression {
            Compression::None => Box::new(Cursor::new(self.bytes)),
            Compression::Gzip => Box::new(GzDecoder::new(Cursor::new(self.bytes))),
        }
    }

    pub(crate) fn decode(self) -> Result<Vec<u8>, LoadError> {
        let mut output = Vec::new();
        self.into_reader().read_to_end(&mut output)?;
        Ok(output)
    }
}

/// The four assets needed by a Bergamot translation model.
#[derive(Debug)]
pub struct ModelAssets {
    pub model: Asset,
    pub vocabularies: VocabularyAssets,
    pub shortlist: Asset,
}

/// Vocabulary assets used by a translation model.
///
/// Most Mozilla models use one shared `SentencePiece` model. Representing that
/// directly avoids reading, decompressing and parsing the same asset twice.
#[derive(Debug)]
pub enum VocabularyAssets {
    Shared(Asset),
    Split { source: Asset, target: Asset },
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;

    use super::Asset;

    #[test]
    fn raw_asset_decodes_original_bytes() {
        assert_eq!(Asset::raw(b"model".to_vec()).decode().unwrap(), b"model");
    }

    #[test]
    fn gzip_asset_decompresses_bytes() {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(b"compressed model").unwrap();
        let bytes = encoder.finish().unwrap();
        assert_eq!(Asset::gzip(bytes).decode().unwrap(), b"compressed model");
    }

    #[test]
    fn invalid_gzip_returns_error() {
        assert!(Asset::gzip(b"not gzip".to_vec()).decode().is_err());
    }
}
