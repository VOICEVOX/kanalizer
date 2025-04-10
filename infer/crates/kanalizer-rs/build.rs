use std::path::{Path, PathBuf};

static MODEL_TAG: &str = "v1";

fn main() -> anyhow::Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=models/model-c2k.safetensors");
    println!("cargo:rerun-if-changed=models/model-c2k.safetensors.br");

    prepare_model()?;

    Ok(())
}

fn prepare_model() -> anyhow::Result<()> {
    let local_model_path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("./models/model-c2k.safetensors");

    let model_path = if local_model_path.try_exists()? {
        local_model_path
    } else {
        prepare_huggingface_model()?
    };

    prepare_compressed_model(&model_path)?;

    println!(
        "cargo:rustc-env=KANALIZER_MODEL_ROOT={}",
        model_path.parent().unwrap().display()
    );

    Ok(())
}

fn prepare_compressed_model(model_path: &Path) -> anyhow::Result<()> {
    let compressed_model_path = model_path.with_extra_extension("br");
    let is_compressed_model_up_to_date = compressed_model_path.try_exists()? && {
        let compressed_model_modified = compressed_model_path.metadata()?.modified()?;
        let model_modified = model_path.metadata()?.modified()?;
        compressed_model_modified >= model_modified
    };

    if !is_compressed_model_up_to_date {
        compress_model(model_path)?;
    }

    Ok(())
}

fn prepare_huggingface_model() -> anyhow::Result<PathBuf> {
    let model_root = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("models");
    std::fs::create_dir_all(&model_root)?;

    let model_version_path = model_root.join("version.txt");
    let model_path = model_root.join("model-c2k.safetensors");

    let latest_model_exists = model_version_path
        .try_exists()?
        .then(|| std::fs::read_to_string(&model_version_path))
        .transpose()?
        .as_deref()
        == Some(MODEL_TAG);

    if !latest_model_exists {
        download_to(
            &format!(
                "https://huggingface.co/VOICEVOX/kanalizer/resolve/{MODEL_TAG}/model/c2k.safetensors"
            ),
            &model_path,
        )?;

        std::fs::write(&model_version_path, MODEL_TAG)?;
    }

    Ok(model_path)
}

fn download_to(url: &str, path: &Path) -> anyhow::Result<()> {
    static NUM_ATTEMPTS: usize = 6;

    for i in 0..NUM_ATTEMPTS {
        let success = download(url, path)?;
        if success {
            return Ok(());
        }
        std::thread::sleep(std::time::Duration::from_secs((i + 1) as u64 * 5));
    }

    return Err(anyhow::anyhow!(
        "Failed to download model after {NUM_ATTEMPTS} attempts"
    ));

    fn download(url: &str, path: &Path) -> anyhow::Result<bool> {
        let mut file = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
        let response = ureq::get(url)
            .config()
            .http_status_as_error(false)
            .build()
            .call()?;
        match response.status().as_u16() {
            200 => {
                std::io::copy(&mut response.into_body().into_reader(), &mut file)?;
                if path.try_exists()? {
                    std::fs::remove_file(path)?;
                }
                file.persist(path)?;
                Ok(true)
            }
            404 => Err(anyhow::anyhow!("Model not found: {url}")),
            status => {
                let body = response.into_body().read_to_string()?;
                eprintln!("Failed to download model: {status} {body:?}");
                Ok(false)
            }
        }
    }
}

fn compress_model(path: &Path) -> anyhow::Result<()> {
    let output_path = path.with_extra_extension("br");

    let mut input = std::fs::File::open(path)?;
    let mut output = std::fs::File::create(output_path.with_extra_extension("tmp"))?;
    let mut output_writer = brotli::CompressorWriter::new(&mut output, 4096, 11, 22);
    std::io::copy(&mut input, &mut output_writer)?;
    drop(output_writer);

    if output_path.try_exists()? {
        std::fs::remove_file(&output_path)?;
    }
    std::fs::rename(output_path.with_extra_extension("tmp"), &output_path)?;

    Ok(())
}

trait AddExtensionExt {
    fn with_extra_extension(&self, ext: &str) -> PathBuf;
}

impl AddExtensionExt for Path {
    fn with_extra_extension(&self, ext: &str) -> PathBuf {
        self.with_file_name(format!(
            "{}.{}",
            self.file_name().unwrap().to_str().unwrap(),
            ext
        ))
    }
}
