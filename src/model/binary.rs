use std::collections::HashMap;
use std::io::Read;

use sha2::{Digest, Sha256};

use crate::asset::Asset;
use crate::error::LoadError;
use crate::model::ModelConfig;

const BINARY_FILE_VERSION: u64 = 1;
const TYPE_INT8: u64 = 0x0101;
const TYPE_FLOAT32: u64 = 0x0404;
const TYPE_INTGEMM8: u64 = 0x4101;

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TensorType {
    Int8,
    Float32,
    IntGemm8,
}

#[derive(Debug)]
pub(crate) enum TensorData {
    Bytes,
    F32(Vec<f32>),
    QuantizedI8 { values: Vec<i8>, multiplier: f32 },
}

impl TensorData {
    #[cfg(test)]
    #[must_use]
    fn tensor_type(&self) -> TensorType {
        match self {
            Self::Bytes => TensorType::Int8,
            Self::F32(_) => TensorType::Float32,
            Self::QuantizedI8 { .. } => TensorType::IntGemm8,
        }
    }
}

#[derive(Debug)]
pub(crate) struct Tensor {
    pub(crate) name: String,
    pub(crate) shape: Vec<usize>,
    pub(crate) data: TensorData,
}

impl Tensor {
    #[must_use]
    pub(crate) fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    #[cfg(test)]
    #[must_use]
    fn tensor_type(&self) -> TensorType {
        self.data.tensor_type()
    }
}

/// Parsed Marian binary archive before its tensors are consumed and packed by
/// the inference runtime.
#[derive(Debug)]
pub(crate) struct ModelArchive {
    pub(crate) config: ModelConfig,
    tensors: HashMap<String, Tensor>,
}

#[derive(Debug)]
struct Header {
    name_len: usize,
    type_code: u64,
    shape_len: usize,
    data_len: usize,
}

impl ModelArchive {
    /// Parse a Marian v1 binary archive.
    ///
    /// # Errors
    ///
    /// Returns an error if decompression, integrity verification, binary
    /// parsing or basic configuration validation fails.
    pub(crate) fn load(asset: Asset, expected_sha256: Option<[u8; 32]>) -> Result<Self, LoadError> {
        let mut reader = HashingReader::new(asset.into_reader());

        let version = read_u64(&mut reader)?;
        if version != BINARY_FILE_VERSION {
            return Err(LoadError::InvalidModel(format!(
                "unsupported binary version {version}"
            )));
        }

        let headers = read_headers(&mut reader)?;
        let names = read_names(&mut reader, &headers)?;
        let shapes = read_shapes(&mut reader, &headers)?;

        let padding = usize_from_u64(read_u64(&mut reader)?, "header padding")?;
        copy_to_sink(&mut reader, padding)?;

        let mut tensors = HashMap::with_capacity(headers.len());
        let mut config = None;
        for ((header, name), shape) in headers.iter().zip(names).zip(shapes) {
            let mut payload = vec![0; header.data_len];
            reader.read_exact(&mut payload)?;

            if name == "special:model.yml" {
                let yaml = trim_nul(&payload);
                let yaml = std::str::from_utf8(yaml).map_err(|_| {
                    LoadError::InvalidModel("special:model.yml is not UTF-8".into())
                })?;
                config = Some(ModelConfig::parse(yaml)?);
                continue;
            }

            let tensor = parse_tensor(header, name.clone(), shape, payload)?;
            if tensors.insert(name.clone(), tensor).is_some() {
                return Err(LoadError::InvalidModel(format!("duplicate tensor {name}")));
            }
        }

        // Ensure the gzip stream has reached EOF and include any trailer bytes
        // in integrity verification performed by the decoder.
        let mut trailing = [0; 1];
        if reader.read(&mut trailing)? != 0 {
            return Err(LoadError::InvalidModel(
                "trailing bytes after Marian payload".into(),
            ));
        }

        let actual_hash: [u8; 32] = reader.finalize().into();
        if let Some(expected) = expected_sha256
            && actual_hash != expected
        {
            return Err(LoadError::HashMismatch {
                expected: hex(&expected),
                actual: hex(&actual_hash),
            });
        }

        let config = config
            .ok_or_else(|| LoadError::InvalidModel("missing special:model.yml metadata".into()))?;
        config.validate_well_formed()?;

        Ok(Self { config, tensors })
    }

    #[cfg(test)]
    fn tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    #[cfg(test)]
    fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    pub(crate) fn take_tensor(&mut self, name: &str) -> Option<Tensor> {
        self.tensors.remove(name)
    }
}

fn read_headers(reader: &mut impl Read) -> Result<Vec<Header>, LoadError> {
    let item_count = usize_from_u64(read_u64(reader)?, "item count")?;
    if item_count == 0 || item_count > 100_000 {
        return Err(LoadError::InvalidModel(format!(
            "implausible item count {item_count}"
        )));
    }
    (0..item_count)
        .map(|_| {
            Ok(Header {
                name_len: usize_from_u64(read_u64(reader)?, "name length")?,
                type_code: read_u64(reader)?,
                shape_len: usize_from_u64(read_u64(reader)?, "shape length")?,
                data_len: usize_from_u64(read_u64(reader)?, "data length")?,
            })
        })
        .collect()
}

fn read_names(reader: &mut impl Read, headers: &[Header]) -> Result<Vec<String>, LoadError> {
    headers
        .iter()
        .map(|header| {
            if header.name_len == 0 || header.name_len > 16 * 1024 {
                return Err(LoadError::InvalidModel(format!(
                    "invalid tensor name length {}",
                    header.name_len
                )));
            }
            let mut bytes = vec![0; header.name_len];
            reader.read_exact(&mut bytes)?;
            if bytes.pop() != Some(0) {
                return Err(LoadError::InvalidModel(
                    "tensor name is not NUL terminated".into(),
                ));
            }
            String::from_utf8(bytes)
                .map_err(|_| LoadError::InvalidModel("tensor name is not valid UTF-8".into()))
        })
        .collect()
}

fn read_shapes(reader: &mut impl Read, headers: &[Header]) -> Result<Vec<Vec<usize>>, LoadError> {
    headers
        .iter()
        .map(|header| {
            if header.shape_len == 0 || header.shape_len > 8 {
                return Err(LoadError::InvalidModel(format!(
                    "invalid shape rank {}",
                    header.shape_len
                )));
            }
            (0..header.shape_len)
                .map(|_| {
                    let dim = read_i32(reader)?;
                    if dim <= 0 {
                        return Err(LoadError::InvalidModel(format!(
                            "invalid tensor dimension {dim}"
                        )));
                    }
                    usize::try_from(dim).map_err(|_| {
                        LoadError::InvalidModel(format!("invalid tensor dimension {dim}"))
                    })
                })
                .collect()
        })
        .collect()
}

fn parse_tensor(
    header: &Header,
    name: String,
    shape: Vec<usize>,
    payload: Vec<u8>,
) -> Result<Tensor, LoadError> {
    let element_count = checked_elements(&shape)?;
    let data = match header.type_code {
        TYPE_INT8 => {
            ensure_payload(&name, header.data_len, element_count)?;
            TensorData::Bytes
        }
        TYPE_FLOAT32 => {
            let byte_count = element_count
                .checked_mul(4)
                .ok_or_else(|| LoadError::InvalidModel(format!("tensor {name} is too large")))?;
            ensure_payload(&name, header.data_len, byte_count)?;
            let values = payload[..byte_count]
                .chunks_exact(4)
                .map(|chunk| {
                    let bytes = <[u8; 4]>::try_from(chunk).map_err(|_| {
                        LoadError::InvalidModel(format!("tensor {name} has malformed float data"))
                    })?;
                    Ok(f32::from_le_bytes(bytes))
                })
                .collect::<Result<Vec<_>, LoadError>>()?;
            TensorData::F32(values)
        }
        TYPE_INTGEMM8 => {
            let required = element_count
                .checked_add(4)
                .ok_or_else(|| LoadError::InvalidModel(format!("tensor {name} is too large")))?;
            ensure_payload(&name, header.data_len, required)?;
            let values = payload[..element_count]
                .iter()
                .map(|&value| value.cast_signed())
                .collect();
            let multiplier_bytes =
                <[u8; 4]>::try_from(&payload[element_count..required]).map_err(|_| {
                    LoadError::InvalidModel(format!(
                        "tensor {name} has malformed quantization metadata"
                    ))
                })?;
            let multiplier = f32::from_le_bytes(multiplier_bytes);
            if !multiplier.is_finite() || multiplier <= 0.0 {
                return Err(LoadError::InvalidModel(format!(
                    "tensor {name} has invalid quantization multiplier {multiplier}"
                )));
            }
            TensorData::QuantizedI8 { values, multiplier }
        }
        other => {
            return Err(LoadError::InvalidModel(format!(
                "tensor {name} uses unsupported type code {other}"
            )));
        }
    };
    Ok(Tensor { name, shape, data })
}

struct HashingReader<R> {
    inner: R,
    hasher: Sha256,
}

impl<R> HashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
        }
    }

    fn finalize(self) -> sha2::digest::Output<Sha256> {
        self.hasher.finalize()
    }
}

impl<R: Read> Read for HashingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buf)?;
        self.hasher.update(&buf[..count]);
        Ok(count)
    }
}

fn read_u64(reader: &mut impl Read) -> Result<u64, LoadError> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i32(reader: &mut impl Read) -> Result<i32, LoadError> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn usize_from_u64(value: u64, label: &str) -> Result<usize, LoadError> {
    usize::try_from(value)
        .map_err(|_| LoadError::InvalidModel(format!("{label} does not fit usize")))
}

fn copy_to_sink(reader: &mut impl Read, len: usize) -> Result<(), LoadError> {
    let mut remaining = len;
    let mut buffer = [0; 4096];
    while remaining > 0 {
        let chunk = remaining.min(buffer.len());
        reader.read_exact(&mut buffer[..chunk])?;
        remaining -= chunk;
    }
    Ok(())
}

fn checked_elements(shape: &[usize]) -> Result<usize, LoadError> {
    shape.iter().try_fold(1usize, |total, &dim| {
        total
            .checked_mul(dim)
            .ok_or_else(|| LoadError::InvalidModel("tensor shape overflows usize".into()))
    })
}

fn ensure_payload(name: &str, actual: usize, required: usize) -> Result<(), LoadError> {
    // Marian item payloads may include alignment padding after the logical
    // tensor data, hence `actual` is allowed to be larger than `required`.
    if actual < required {
        Err(LoadError::InvalidModel(format!(
            "tensor {name} payload is {actual} bytes, expected at least {required}"
        )))
    } else {
        Ok(())
    }
}

fn trim_nul(bytes: &[u8]) -> &[u8] {
    bytes.strip_suffix(&[0]).unwrap_or(bytes)
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0xf) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;
    use sha2::{Digest, Sha256};

    use crate::asset::Asset;

    use super::{ModelArchive, TYPE_FLOAT32, TYPE_INTGEMM8, TensorData, TensorType};

    struct Item {
        name: String,
        type_code: u64,
        shape: Vec<i32>,
        payload: Vec<u8>,
    }

    fn config_yaml() -> Vec<u8> {
        let mut yaml = br#"
type: transformer
dim-emb: 4
dim-vocabs: [8, 8]
enc-depth: 1
dec-depth: 1
transformer-heads: 2
transformer-dim-ffn: 8
transformer-ffn-depth: 2
transformer-ffn-activation: relu
transformer-decoder-autoreg: rnn
dec-cell: ssru
version: test
"#
        .to_vec();
        yaml.push(0);
        yaml
    }

    fn build(mut items: Vec<Item>, padding: usize) -> Vec<u8> {
        items.insert(
            0,
            Item {
                name: "special:model.yml".into(),
                type_code: 0x0101,
                shape: vec![1],
                payload: config_yaml(),
            },
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for item in &items {
            bytes.extend_from_slice(&((item.name.len() + 1) as u64).to_le_bytes());
            bytes.extend_from_slice(&item.type_code.to_le_bytes());
            bytes.extend_from_slice(&(item.shape.len() as u64).to_le_bytes());
            bytes.extend_from_slice(&(item.payload.len() as u64).to_le_bytes());
        }
        for item in &items {
            bytes.extend_from_slice(item.name.as_bytes());
            bytes.push(0);
        }
        for item in &items {
            for dim in &item.shape {
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&(padding as u64).to_le_bytes());
        bytes.resize(bytes.len() + padding, 0);
        for item in items {
            bytes.extend_from_slice(&item.payload);
        }
        bytes
    }

    fn float_item(name: &str, values: &[f32]) -> Item {
        Item {
            name: name.into(),
            type_code: TYPE_FLOAT32,
            shape: vec![values.len() as i32],
            payload: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }

    #[test]
    fn loads_minimal_raw_archive() {
        let archive = ModelArchive::load(Asset::raw(build(vec![], 0)), None).unwrap();
        assert_eq!(archive.config.version, "test");
        assert_eq!(archive.tensor_count(), 0);
    }

    #[test]
    fn loads_gzip_archive() {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&build(vec![], 0)).unwrap();
        let archive = ModelArchive::load(Asset::gzip(encoder.finish().unwrap()), None).unwrap();
        assert_eq!(archive.config.dim_emb, 4);
    }

    #[test]
    fn parses_float32_tensor() {
        let archive = ModelArchive::load(
            Asset::raw(build(vec![float_item("weight", &[1.5, -2.0])], 0)),
            None,
        )
        .unwrap();
        let TensorData::F32(values) = &archive.tensor("weight").unwrap().data else {
            panic!("expected float tensor")
        };
        assert_eq!(values, &[1.5, -2.0]);
        assert_eq!(
            archive.tensor("weight").unwrap().tensor_type(),
            TensorType::Float32
        );
    }

    #[test]
    fn parses_intgemm8_tensor_and_multiplier() {
        let mut payload = vec![1u8, 255];
        payload.extend_from_slice(&2.5f32.to_le_bytes());
        let archive = ModelArchive::load(
            Asset::raw(build(
                vec![Item {
                    name: "weight".into(),
                    type_code: TYPE_INTGEMM8,
                    shape: vec![2],
                    payload,
                }],
                0,
            )),
            None,
        )
        .unwrap();
        let TensorData::QuantizedI8 { values, multiplier } =
            &archive.tensor("weight").unwrap().data
        else {
            panic!("expected intgemm tensor")
        };
        assert_eq!(values, &[1, -1]);
        assert_eq!(*multiplier, 2.5);
        assert_eq!(
            archive.tensor("weight").unwrap().tensor_type(),
            TensorType::IntGemm8
        );
    }

    #[test]
    fn accepts_header_and_tensor_payload_padding() {
        let mut item = float_item("weight", &[1.0]);
        item.payload.extend_from_slice(&[0; 12]);
        let archive = ModelArchive::load(Asset::raw(build(vec![item], 13)), None).unwrap();
        assert!(archive.tensor("weight").is_some());
    }

    #[test]
    fn rejects_invalid_version_type_and_short_payload() {
        let mut version = build(vec![], 0);
        version[..8].copy_from_slice(&2u64.to_le_bytes());
        assert!(ModelArchive::load(Asset::raw(version), None).is_err());

        let invalid_type = Item {
            name: "weight".into(),
            type_code: 99,
            shape: vec![1],
            payload: vec![0],
        };
        assert!(ModelArchive::load(Asset::raw(build(vec![invalid_type], 0)), None).is_err());

        let short = Item {
            name: "weight".into(),
            type_code: TYPE_FLOAT32,
            shape: vec![2],
            payload: 1.0f32.to_le_bytes().to_vec(),
        };
        assert!(ModelArchive::load(Asset::raw(build(vec![short], 0)), None).is_err());
    }

    #[test]
    fn rejects_duplicate_names_and_trailing_bytes() {
        let duplicate = build(
            vec![float_item("weight", &[1.0]), float_item("weight", &[2.0])],
            0,
        );
        assert!(ModelArchive::load(Asset::raw(duplicate), None).is_err());

        let mut trailing = build(vec![], 0);
        trailing.push(0);
        assert!(ModelArchive::load(Asset::raw(trailing), None).is_err());
    }

    #[test]
    fn verifies_hash_and_rejects_malformed_name_or_shape() {
        let bytes = build(vec![], 0);
        let hash: [u8; 32] = Sha256::digest(&bytes).into();
        ModelArchive::load(Asset::raw(bytes.clone()), Some(hash)).unwrap();
        assert!(ModelArchive::load(Asset::raw(bytes.clone()), Some([0; 32])).is_err());

        let mut bad_name = bytes.clone();
        let names_start = 16 + 32;
        let name_end = names_start + "special:model.yml".len();
        bad_name[name_end] = b'x';
        assert!(ModelArchive::load(Asset::raw(bad_name), None).is_err());

        let mut bad_shape = bytes;
        let shape_start = names_start + "special:model.yml".len() + 1;
        bad_shape[shape_start..shape_start + 4].copy_from_slice(&0i32.to_le_bytes());
        assert!(ModelArchive::load(Asset::raw(bad_shape), None).is_err());
    }
}
