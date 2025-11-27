use age::{x25519, Decryptor, Encryptor};
use anyhow::{anyhow, Result};
use std::fs;
use std::io::{Read, Write};
use std::path::Path;
use std::str::FromStr;

pub fn encrypt_text(text: &str, recipient: &str) -> Result<Vec<u8>> {
    let recipient =
        x25519::Recipient::from_str(recipient.trim()).map_err(|e| anyhow!(e.to_string()))?;
    let encryptor = Encryptor::with_recipients(vec![Box::new(recipient)])
        .ok_or_else(|| anyhow!("missing recipients"))?;
    let mut output = Vec::new();
    let mut writer = encryptor.wrap_output(&mut output)?;
    writer.write_all(text.as_bytes())?;
    writer.finish()?;
    Ok(output)
}

pub fn decrypt_text(ciphertext: &[u8], identity_path: &Path) -> Result<String> {
    let identity_content = fs::read_to_string(identity_path)?;
    let identity_line = identity_content
        .lines()
        .map(|l| l.trim())
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .ok_or_else(|| anyhow!("identity file is empty"))?;
    let identity = x25519::Identity::from_str(identity_line)
        .map_err(|e| anyhow!(format!("invalid identity: {e}")))?;
    let decryptor = match Decryptor::new(ciphertext)? {
        Decryptor::Recipients(d) => d,
        _ => return Err(anyhow!("unsupported decryptor")),
    };
    let identities: Vec<Box<dyn age::Identity>> = vec![Box::new(identity)];
    let mut decrypted = Vec::new();
    let mut reader = decryptor.decrypt(identities.iter().map(|id| id.as_ref()))?;
    reader.read_to_end(&mut decrypted)?;
    Ok(String::from_utf8(decrypted)?)
}
