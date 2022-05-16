use anyhow::{bail, Result};
use std::io::{Cursor, Read, Write};
use tractor::{BasicInferer, DynamicBatchingInferer, FixedBatchingInferer};

/// Magic used to ensure assets are valid.
pub const MAGIC: [u8; 4] = [b'C', b'R', b'V', b'O'];

/// AssetKind denotes what kind of policy is contained inside an [`AssetData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AssetKind {
    /// Used for an asset containing ONNX ModelProto data.
    Onnx = 1,

    /// Used for an asset containing NNEF data.
    Nnef = 2,
}

impl TryFrom<u8> for AssetKind {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(AssetKind::Onnx),
            2 => Ok(AssetKind::Nnef),
            v => bail!("unexpected asset kind: {:?}", v),
        }
    }
}

impl std::fmt::Display for AssetKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AssetKind::Onnx => f.pad("onnx"),
            AssetKind::Nnef => f.pad("nnef"),
        }
    }
}
/// Contains a tagged buffer of policy data.
#[derive(Debug, Clone)]
#[repr(C)]
pub struct AssetData {
    kind: AssetKind,
    data: Vec<u8>,
}

impl AssetData {
    /// Create a new AssetData from parts.
    ///
    /// Note: Does not validate the data.
    pub fn new<Data: Into<Vec<u8>>>(kind: AssetKind, data: Data) -> Self {
        Self {
            kind,
            data: data.into(),
        }
    }

    /// Create a new AssetData from parts.
    ///
    /// Note: Does not validate the data.
    pub fn from_reader<Reader: Read>(kind: AssetKind, mut reader: Reader) -> Result<Self> {
        let mut buf = vec![];
        reader.read_to_end(&mut buf)?;

        Ok(Self::new(kind, buf))
    }

    /// Deserialize from raw bytes.
    ///
    /// Note: Does not validate data; only loads it as an asset. Validation happens when creating an inferer.
    pub fn deserialize(mut reader: impl Read) -> Result<Self> {
        let mut magic: [u8; 4] = [0; 4];
        let count = reader.read(&mut magic)?;
        if count < 4 {
            anyhow::bail!("too few bytes available, expected 4 but got {:?}", count);
        }

        if magic != MAGIC {
            anyhow::bail!(
                "unexpected magic: expected 'CRVO' found {}{}{}{}",
                magic[0] as char,
                magic[1] as char,
                magic[2] as char,
                magic[3] as char
            );
        }

        let mut preamble: [u8; 4] = [0; 4];
        let count = reader.read(&mut preamble)?;
        if count < 4 {
            anyhow::bail!("too few bytes available, expected 4 but got {:?}", count);
        }

        let kind = preamble[3].try_into()?;
        let mut data = vec![];
        reader.read_to_end(&mut data)?;

        Ok(Self { kind, data })
    }

    /// Serialize to raw bytes.
    ///
    /// The buffer returned will not contain any extra unused bytes.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut output = vec![];
        output.write_all(&MAGIC)?;

        let preamble: [u8; 4] = [0, 0, 0, self.kind as u8];
        output.write_all(&preamble)?;

        output.extend(&self.data);
        output.shrink_to_fit();
        Ok(output)
    }

    /// Get the kind of this asset.
    pub fn kind(&self) -> AssetKind {
        self.kind
    }

    /// Get the asset data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Load a simple unbatching inferer from this asset.
    ///
    /// See ['BasicInferer'] for more details.
    pub fn load_simple(&self) -> Result<BasicInferer> {
        let mut cursor = Cursor::new(&self.data);
        match self.kind {
            AssetKind::Onnx => tractor_onnx::simple_inferer_from_stream(&mut cursor),
            AssetKind::Nnef => tractor_nnef::simple_inferer_from_stream(&mut cursor),
        }
    }

    /// Load a batching inferer from this asset with fixed batch sizes.
    ///
    /// See [`FixedBatchingInferer`] for more details.
    pub fn load_fixed_batcher(&self, sizes: &[usize]) -> Result<FixedBatchingInferer> {
        let mut cursor = Cursor::new(&self.data);
        match self.kind {
            AssetKind::Onnx => tractor_onnx::fixed_batch_inferer_from_stream(&mut cursor, sizes),
            AssetKind::Nnef => tractor_nnef::fixed_batch_inferer_from_stream(&mut cursor, sizes),
        }
    }

    /// Load a batching inferer from this asset with dynamic batch sizes.
    ///
    /// See [`DynamicBatchingInferer`] for more details.
    pub fn load_dynamic_batcher(&self, sizes: &[usize]) -> Result<DynamicBatchingInferer> {
        let mut cursor = Cursor::new(&self.data);
        match self.kind {
            AssetKind::Onnx => tractor_onnx::batched_inferer_from_stream(&mut cursor, sizes),
            AssetKind::Nnef => tractor_nnef::batched_inferer_from_stream(&mut cursor, sizes),
        }
    }

    /// Convert this to an NNEF asset.
    ///
    /// Will return an error if this is already an NNEF asset.
    pub fn as_nnef(&self, batch_size: Option<usize>) -> Result<Self> {
        if self.kind == AssetKind::Nnef {
            bail!("trying to convert from nnef to nnef");
        }

        let mut cursor = Cursor::new(&self.data);
        let data = tractor_onnx::to_nnef(&mut cursor, batch_size)?;

        Ok(Self {
            data,
            kind: AssetKind::Nnef,
        })
    }
}
