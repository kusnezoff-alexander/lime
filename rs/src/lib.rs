use eggmock::{egg::{EGraph, Id}, Mig, MigLanguage, RewriteFFI, Rewriter, RewriterResult};

struct LimeMigRewriter;

impl Rewriter for LimeMigRewriter {
    type Network = Mig;
    type Analysis = ();

    fn create_analysis(&mut self) -> Self::Analysis {
        ()
    }

    fn rewrite(
        &mut self,
        _egraph: EGraph<MigLanguage, Self::Analysis>,
        _roots: impl Iterator<Item = Id>,
    ) -> RewriterResult<Mig> {
        todo!()
    }
}

#[no_mangle]
pub extern "C" fn lime_rewrite() -> RewriteFFI<Mig> {
    RewriteFFI::new(LimeMigRewriter)
}
