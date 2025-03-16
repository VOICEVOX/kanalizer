use pyo3::prelude::*;
use pyo3::types::PyDict;

fn extract_strategy(strategy: &str, kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<e2k::Strategy> {
    Ok(match strategy {
        "greedy" => e2k::Strategy::Greedy,
        "top_k" => {
            let k = kwargs
                .map(|kwargs| kwargs.get_item("k"))
                .transpose()?
                .flatten()
                .map(|k| k.extract::<usize>())
                .transpose()?;

            let mut strategy = e2k::StrategyTopK::default();
            if let Some(k) = k {
                strategy.k = k;
            }

            e2k::Strategy::TopK(strategy)
        }
        "top_p" => {
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

            let mut strategy = e2k::StrategyTopP::default();
            if let Some(top_p) = top_p {
                strategy.top_p = top_p;
            }
            if let Some(temperature) = temperature {
                strategy.temperature = temperature;
            }

            e2k::Strategy::TopP(strategy)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "strategy must be one of 'greedy', 'top_k', 'top_p'",
            ));
        }
    })
}

#[pyclass(frozen)]
struct C2k {
    inner: std::sync::RwLock<e2k::C2k>,
}

#[pymethods]
impl C2k {
    #[new]
    #[pyo3(signature = (max_length = 32))]
    fn new(max_length: usize) -> Self {
        Self {
            inner: std::sync::RwLock::new(e2k::C2k::new(max_length)),
        }
    }

    #[pyo3(signature = (strategy, **kwargs))]
    fn set_decode_strategy(
        &self,
        strategy: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let strategy = extract_strategy(strategy, kwargs)?;

        {
            let mut inner = self.inner.write().unwrap();

            inner.set_decode_strategy(strategy);
        };

        Ok(())
    }

    fn __call__(&self, src: &str) -> String {
        self.inner.read().unwrap().infer(src)
    }
}

#[pymodule(name = "voicevox_e2k")]
fn voicevox_e2k(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<C2k>()?;

    m.add("KANAS", e2k::KANAS)?;
    m.add("ASCII_ENTRIES", e2k::ASCII_ENTRIES)?;

    Ok(())
}
