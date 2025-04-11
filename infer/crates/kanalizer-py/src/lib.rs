use pyo3::prelude::*;
use pyo3::types::PyDict;

fn extract_strategy(
    strategy: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<kanalizer::Strategy> {
    return Ok(match strategy {
        "greedy" => {
            error_on_extra_args(kwargs, &[])?;

            kanalizer::Strategy::Greedy
        }
        "top_k" => {
            error_on_extra_args(kwargs, &["k"])?;

            let k = kwargs
                .map(|kwargs| kwargs.get_item("k"))
                .transpose()?
                .flatten()
                .map(|k| k.extract::<usize>())
                .transpose()?;

            let mut strategy = kanalizer::StrategyTopK::default();
            if let Some(k) = k {
                strategy.k = k;
            }

            kanalizer::Strategy::TopK(strategy)
        }
        "top_p" => {
            error_on_extra_args(kwargs, &["p", "t"])?;

            let top_p = kwargs
                .map(|kwargs| kwargs.get_item("p"))
                .transpose()?
                .flatten()
                .map(|top_p| top_p.extract::<f32>())
                .transpose()?;

            let temperature = kwargs
                .map(|kwargs| kwargs.get_item("t"))
                .transpose()?
                .flatten()
                .map(|temperature| temperature.extract::<f32>())
                .transpose()?;

            let mut strategy = kanalizer::StrategyTopP::default();
            if let Some(top_p) = top_p {
                strategy.top_p = top_p;
            }
            if let Some(temperature) = temperature {
                strategy.temperature = temperature;
            }

            kanalizer::Strategy::TopP(strategy)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "strategy must be one of 'greedy', 'top_k', 'top_p'",
            ));
        }
    });

    fn error_on_extra_args(kwargs: Option<&Bound<'_, PyDict>>, expected: &[&str]) -> PyResult<()> {
        if let Some(kwargs) = kwargs {
            let keys = kwargs.iter().map(|item| item.0).collect::<Vec<_>>();
            let keys = keys
                .iter()
                .map(|key| key.extract::<String>())
                .collect::<Result<Vec<_>, _>>()?;

            let extra_keys = keys
                .iter()
                .map(|key| key.as_str())
                .filter(|&key| !expected.contains(&key))
                .collect::<Vec<_>>();

            if !extra_keys.is_empty() {
                return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "unexpected keyword argument(s): {}",
                    extra_keys.join(", ")
                )));
            }
        }
        Ok(())
    }
}

pyo3::import_exception!(kanalizer._error, IncompleteConversionError);

#[pyfunction]
#[pyo3(signature = (word, /, *, max_length = 32, strict = true, error_on_incomplete = true, strategy = "greedy", **kwargs))]
fn convert(
    word: &str,
    max_length: usize,
    strict: bool,
    error_on_incomplete: bool,
    strategy: &str,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<String> {
    let strategy = extract_strategy(strategy, kwargs)?;
    let result = kanalizer::convert(word)
        .with_max_length(max_length.try_into().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("max_length must be a positive integer")
        })?)
        .with_strategy(&strategy)
        .with_strict(strict)
        .with_error_on_incomplete(error_on_incomplete)
        .perform();

    match result {
        Ok(dst) => Ok(dst),
        Err(err @ (kanalizer::Error::EmptyInput | kanalizer::Error::InvalidChars { .. })) => {
            Err(pyo3::exceptions::PyValueError::new_err(err.to_string()))
        }
        Err(
            ref err @ kanalizer::Error::IncompleteConversion {
                ref incomplete_output,
            },
        ) => Err(IncompleteConversionError::new_err((
            err.to_string(),
            incomplete_output.to_owned(),
        ))),
    }
}

#[pymodule(name = "_rust")]
fn init_kanalizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("INPUT_CHARS", &*kanalizer::INPUT_CHARS)?;
    m.add("OUTPUT_CHARS", &*kanalizer::OUTPUT_CHARS)?;
    m.add_function(wrap_pyfunction!(convert, m)?)?;

    Ok(())
}
