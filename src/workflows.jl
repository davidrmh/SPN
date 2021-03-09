include("spn.jl");
#======================================================
Workflow for MLE in selective SPNs (Fixed architecture)
======================================================#
#True data generator process
regsel = regularselective();
nsamples = 20_000;
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
nsamples = 100;
modeldata = sample(model, nsamples);
modelparam = getparameters(model, false)[1];
logpdfmem = Dict();
logpdf!(model, modeldata, modelparam, logpdfmem);
#nodes derivatives
nodesderiv = nodesderivatives(model, logpdfmem, true);
weights_deriv = weights_loglike_deriv(model, logpdfmem, nodesderiv);
#======================================================
Workflow for optimising weights using gradient ascent
PENDING WORK
======================================================#
_reset_counter!();
realmodel = regularselective();
nsamples = 1_000;
data = sample(realmodel, nsamples);
realparam = getparameters(realmodel, false)[1];
reallogpdf = sum(logpdf(realmodel, data, realparam));
_reset_counter!();
model = regularselective();
initializeweights!(model);
niter = 10;
rate = 0.05;
learn_weights_gradient!(model, data, niter, rate, false);
modelparam = getparameters(model, false)[1];
modellogpdf = sum(logpdf(model, data, modelparam));
dotfile(model, "dotfiles\\test.gv");
#=============================================================
Workflow for optimising weights using Expectation Maximization
=============================================================#
_reset_counter!();
realmodel = regularselective();
nsamples = 10000;
data = sample(realmodel, nsamples);
realparam = getparameters(realmodel, false)[1];
reallogpdf = sum(logpdf(realmodel, data, realparam));
_reset_counter!();
model = regularselective();
initializeweights!(model);
niter = 100;
learn_weights_em!(model, data, niter, false);
modelparam = getparameters(model, false)[1];
modellogpdf = sum(logpdf(model, data, modelparam));
dotfile(model, "dotfiles\\test.gv");
