include("spn.jl");
#======================================================
Workflow for MLE in selective SPNs (Fixed architecture)
======================================================#
#True data generator process
regsel = regularselective();
nsamples = 10_000;
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
#======================================================
Workflow for calculating the partial derivatives of
the log-likelihood function with respect to the weights
of each sum node
======================================================#
_reset_counter!();
model = regularselective();
nsamples = 1;
modeldata = sample(model, nsamples);
modelparam = getparameters(model, false)[1];
logpdfmem = Dict();
logpdf!(model, modeldata, modelparam, logpdfmem);
#nodes derivatives
nodesderiv = nodesderivatives(model, logpdfmem, true);
weights_deriv = weights_loglike_deriv(model, logpdfmem, nodesderiv);
