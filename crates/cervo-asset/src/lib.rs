use anyhow::{bail, Result};
use std::io::{Cursor, Read, Write};
use tractor::{BasicInferer, DynamicBatchingInferer, FixedBatchingInferer};

/// Magic used to ensure assets are valid.
pub const MAGIC: [u8; 4] = ['C' as u8, 'R' as u8, 'V' as u8, 'O' as u8];

/// AssetKind denotes what kind of policy is contained inside an [`AssetData`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AssetKind {
    /// Used for an asset containing ONNX ModelProto data.
    Onnx = 1,
}

impl TryFrom<u8> for AssetKind {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(AssetKind::Onnx),
            v => bail!("unexpected asset kind: {:?}", v),
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

    /// Deserialize from raw bytes.
    ///
    /// Note: Does not validate data; only loads it as an asset. Validation happens when creating an inferer.
    pub fn deserialize(reader: &mut dyn Read) -> Result<Self> {
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
        output.write(&MAGIC)?;

        let preamble: [u8; 4] = [0, 0, 0, self.kind as u8];
        output.write(&preamble)?;

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
        }
    }

    /// Load a batching inferer from this asset with fixed batch sizes.
    ///
    /// See [`FixedBatchingInferer`] for more details.
    pub fn load_fixed_batcher(&self, sizes: &[usize]) -> Result<FixedBatchingInferer> {
        let mut cursor = Cursor::new(&self.data);
        match self.kind {
            AssetKind::Onnx => tractor_onnx::fixed_batch_inferer_from_stream(&mut cursor, sizes),
        }
    }

    /// Load a batching inferer from this asset with fixed batch sizes.
    ///
    /// See [`DynamicBatchingInferer`] for more details.
    pub fn load_dynamic_batcher(&self, sizes: &[usize]) -> Result<DynamicBatchingInferer> {
        let mut cursor = Cursor::new(&self.data);
        match self.kind {
            AssetKind::Onnx => tractor_onnx::batched_inferer_from_stream(&mut cursor, sizes),
        }
    }

    /// Convert this to an NNEF asset.
    pub fn as_nnef(&self) -> Self {
        match self.kind {
            // NOTE(TSolberg): Actually a lie for debugging purposes; waiting on https://github.com/EmbarkStudios/tractor/pull/2
            AssetKind::Onnx => Self {
                kind: AssetKind::Onnx,
                data: self.data.clone(),
            },
        }
    }
}
