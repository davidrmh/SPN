include("spn.jl");
#======================================================
Workflow for MLE in selective SPNs (Fixed architecture)
======================================================#
#True data generator process
regsel = regularselective();
nsamples = 100_000;
data = sample(regsel, nsamples);

_reset_counter!();
model = regularselective();
modelparams = getparameters(model, false)[1];
dictval = Dict();
#Update dictval
logpdf!(model, data, modelparams, dictval);
#Get calculation trees
calctrees = getcalctrees(model, dictval);
#Update model's parameters
selective_mle!(model, calctrees);
dotfile(model, "dotfiles\\test.gv");
