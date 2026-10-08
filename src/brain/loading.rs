//! Ordered parallel transforms for data loaders. Readers remain sequential;
//! independent records are processed in bounded batches on native targets.

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

pub(super) fn map_ordered<T: Send, U: Send>(
    items: Vec<T>,
    transform: impl Fn(T) -> U + Send + Sync,
) -> Vec<U> {
    #[cfg(not(target_arch = "wasm32"))]
    return items.into_par_iter().map(transform).collect();
    #[cfg(target_arch = "wasm32")]
    items.into_iter().map(transform).collect()
}

pub(super) fn map_batches<T: Send, U: Send>(
    items: impl Iterator<Item = T>,
    transform: impl Fn(T) -> U + Send + Sync,
) -> Vec<U> {
    let mut items = items;
    let mut output = Vec::new();
    loop {
        let batch: Vec<_> = items.by_ref().take(1024).collect();
        if batch.is_empty() {
            break;
        }
        output.extend(map_ordered(batch, &transform));
    }
    output
}
