//! Opt-in, synchronous numerical diagnostics. No tensor readbacks when disabled.
use std::{cell::RefCell, fs::{File, OpenOptions}, io::Write, path::Path};
use burn::{prelude::*, module::{ModuleVisitor, Param}, optim::GradientsParams,
    tensor::backend::AutodiffBackend};

struct State { file: File, context: String, active: bool }
thread_local! { static STATE: RefCell<Option<State>> = const { RefCell::new(None) }; }

pub struct Session;
impl Session {
    pub fn start(dir: &Path) -> anyhow::Result<Self> {
        if std::env::var("YUMON_TRAIN_DEBUG").as_deref() == Ok("1") {
            let path = dir.join("training-debug.log");
            let file = OpenOptions::new().create(true).append(true).open(&path)?;
            STATE.with(|s| *s.borrow_mut() = Some(State { file, context: String::new(), active: false }));
            event(&format!("session start utc={} pid={} log={} (expensive tensor checks enabled)",
                chrono::Utc::now(), std::process::id(), path.display()));
        }
        Ok(Self)
    }
}
impl Drop for Session { fn drop(&mut self) { STATE.with(|s| *s.borrow_mut() = None); } }

pub fn event(message: &str) {
    STATE.with(|s| {
        if let Some(s) = s.borrow_mut().as_mut() {
            let line = format!("[train-debug] {} {message}\n", s.context);
            eprint!("{line}");
            if let Err(e) = s.file.write_all(line.as_bytes()).and_then(|_| s.file.flush()) {
                eprintln!("[train-debug] log write failed: {e}");
            }
        }
    });
}
pub fn active() -> bool { STATE.with(|s| s.borrow().as_ref().is_some_and(|s| s.active)) }

pub struct Batch;
impl Batch {
    pub fn start(context: String) -> Self {
        STATE.with(|s| { if let Some(s) = s.borrow_mut().as_mut() { s.context = context; s.active = true; } });
        event("batch start");
        Self
    }
    pub fn finish(self) {
        event("batch complete");
        STATE.with(|s| { if let Some(s) = s.borrow_mut().as_mut() { s.active = false; } });
    }
}
impl Drop for Batch {
    fn drop(&mut self) {
        if active() { event("ABORT: batch did not complete; see last phase/check and stderr error"); }
        STATE.with(|s| { if let Some(s) = s.borrow_mut().as_mut() { s.active = false; } });
    }
}

/// Reduce on the GPU, read only small summaries. Never modify the training graph.
pub fn check<B: Backend, const D: usize>(label: &str, x: &Tensor<B, D>) {
    if !active() { return; }
    event(&format!("checking {label} shape={:?} dtype={:?}", x.dims(), x.dtype()));
    let finite = x.clone().detach().is_finite().all().into_scalar();
    if !finite {
        event(&format!("NON-FINITE {label}"));
        panic!("training-debug: non-finite {label}; see training-debug.log");
    }
    let max = x.clone().detach().abs().max().into_data().convert::<f32>().to_vec::<f32>().unwrap()[0];
    event(&format!("{label} finite=true max_abs={max:.6e}"));
}

/// One boolean GPU reduction per parameter; print only failures and a summary.
pub fn check_parameters<B: AutodiffBackend, M: Module<B>>(model: &M, grads: Option<&GradientsParams>) {
    if !active() { return; }
    struct Visitor<'a> { grads: Option<&'a GradientsParams>, count: usize, path: Vec<String> }
    impl<B: AutodiffBackend> ModuleVisitor<B> for Visitor<'_> {
        fn enter_module(&mut self, name: &str, _: &str) { self.path.push(name.into()); }
        fn exit_module(&mut self, _: &str, _: &str) { self.path.pop(); }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
            let value = match self.grads {
                Some(grads) => match grads.get::<B::InnerBackend, D>(p.id) { Some(g) => g, None => return },
                None => p.val().inner(),
            };
            if !value.clone().is_finite().all().into_scalar() {
                let label = format!("{} path={} parameter={:?}",
                    if self.grads.is_some() { "gradient" } else { "weight" }, self.path.join("."), p.id);
                check(&label, &value);
            }
            self.count += 1;
        }
    }
    let label = if grads.is_some() { "gradients before optimizer" } else { "weights after optimizer" };
    event(&format!("checking {label}"));
    let mut visitor = Visitor { grads, count: 0, path: vec![] };
    model.visit(&mut visitor);
    event(&format!("{label}: {} tensors finite", visitor.count));
}
