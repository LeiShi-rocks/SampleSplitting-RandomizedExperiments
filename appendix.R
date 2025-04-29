# Compute estimators
## model fitting + estimator
part_fit = function(dat, model_id, est_id, method = "lm", calibration = FALSE, model_opts = list()){
  # grab all names starting with "X"
  N = nrow(dat)
  if(is.null(model_opts$xvars)){
    xvars = grep("^X", names(dat), value = TRUE)
  }
  else{
    xvars = model_opts$xvars
  }
  
  # run linear regression with model_id
  if (method == "lm"){
    # (a) via paste/as.formula:
    form = as.formula(
      paste0("Y ~ ", paste(xvars, collapse = " + "))
    )
    
    fXZ = lm(form, data = dat %>% filter(model_id == 1 & Z == 1))
    fXC = lm(form, data = dat %>% filter(model_id == 1 & Z == 0))
    
    # construct estimator with est_id
    ypredZ = predict(fXZ, newdata = dat)
    ypredC = predict(fXC, newdata = dat)
  }
  else if (method == "gam"){
    # GAM 
    ## build a GAM formula: Y ~ s(X1) + s(X2) + ... + s(Xd)
    smooth_terms = paste0("s(", xvars, ", bs = 'cs', k = 7)", collapse = " + ")
    form_gam = as.formula(paste("Y ~", smooth_terms))
    
    # fit a gam with rows: model_id == 1 & Z == 1
    fXZ = mgcv::gam(
      form_gam,
      data   = dat %>% dplyr::filter(model_id == 1 & Z == 1),
      method = "REML"          # (REML gives more stable smoothing‐parameter estimates)
    )
    
    # fit a gam with rows: model_id == 1 & Z == 0
    fXC = mgcv::gam(
      form_gam,
      data   = dat %>% dplyr::filter(model_id == 1 & Z == 0),
      method = "REML"
    ) 
    
    # prediction
    ypredZ = predict(fXZ, newdata = dat, type = "response")
    ypredC = predict(fXC, newdata = dat, type = "response")
    
  }
  else if (method == "rf") {
    ## build a formula Y ~ X1 + X2 + … + Xd
    form_rf <- as.formula(
      paste("Y ~", paste(xvars, collapse = " + "))
    )
    
    ## fit separate forests for treated and control groups
    Z_id = (model_id == 1) & (dat$Z == 1)
    C_id = (model_id == 1) & (dat$Z == 0)
    fXZ = randomForest::randomForest(
      form_rf,
      data   = dat %>% dplyr::filter(Z_id),
      nodesize = 20
    )
    
    fXC = randomForest::randomForest(
      form_rf,
      data   = dat %>% dplyr::filter(C_id),
      nodesize = 20
    )
    
    ypredZ = rep(NA, N)
    ypredC = rep(NA, N)

    ypredZ[Z_id] = stats::predict(fXZ)
    ypredZ[C_id] = stats::predict(fXZ, newdata = dat[C_id,])
    ypredZ[(!C_id) & (!Z_id)] = stats::predict(fXZ, newdata = dat[(!C_id) & (!Z_id),])

    ypredC[Z_id] = stats::predict(fXC, newdata = dat[Z_id,])
    ypredC[C_id] = stats::predict(fXC)
    ypredC[(!C_id) & (!Z_id)] = stats::predict(fXC, newdata = dat[(!C_id) & (!Z_id),])
    
    ## predictions on the full data set - Out-of-bag prediction vs. full-tree prediction
    # ypredZ2 = stats::predict(fXZ, newdata = dat)
    # ypredC2 = stats::predict(fXC, newdata = dat)
    
    
  }
  else if (method == "poisson") {
    if (any(dat$Y < 0 | abs(dat$Y - round(dat$Y)) > 1e-8))
      warning("Y is not an (approximate) count; Poisson GLM may be inappropriate.")
    
    Z_id = (model_id == 1) & (dat$Z == 1)
    C_id = (model_id == 1) & (dat$Z == 0)
    
    ## plain formula: Y ~ X1 + … + Xd (same as lm)
    form_pois <- as.formula(
      paste("Y ~", paste(xvars, collapse = " + "))
    )
    
    ## fit separate Poisson regressions for treated & control rows
    fXZ <- glm(
      form_pois,
      family = "poisson",
      data   = dat %>% dplyr::filter(Z_id)
    )
    
    fXC <- glm(
      form_pois,
      family = "poisson",
      data   = dat %>% dplyr::filter(C_id)
    )
    
    ypredZ = rep(NA, N)
    ypredC = rep(NA, N)
    
    ypredZ[Z_id] = stats::predict(fXZ, type = "response")
    ypredZ[C_id] = stats::predict(fXZ, newdata = dat[C_id,], type = "response")
    ypredZ[(!C_id) & (!Z_id)] = stats::predict(fXZ, newdata = dat[(!C_id) & (!Z_id),], type = "response")
    
    ypredC[Z_id] = stats::predict(fXC, newdata = dat[Z_id,], type = "response")
    ypredC[C_id] = stats::predict(fXC, type = "response")
    ypredC[(!C_id) & (!Z_id)] = stats::predict(fXC, newdata = dat[(!C_id) & (!Z_id),], type = "response")
    
    ## mean‑scale predictions on the full data set
    # ypredZ <- stats::predict(fXZ, newdata = dat, type = "response")
    # ypredC <- stats::predict(fXC, newdata = dat, type = "response")
    
  }
  else if (method == "binomial") {
    Z_id = (model_id == 1) & (dat$Z == 1)
    C_id = (model_id == 1) & (dat$Z == 0)
    
    ## plain formula: Y ~ X1 + … + Xd (same as lm)
    form_binomial <- as.formula(
      paste("Y ~", paste(xvars, collapse = " + "))
    )
    
    ## fit separate Poisson regressions for treated & control rows
    fXZ <- glm(
      form_binomial,
      family = "binomial",
      data   = dat %>% dplyr::filter(Z_id)
    )
    
    fXC <- glm(
      form_binomial,
      family = "binomial",
      data   = dat %>% dplyr::filter(C_id)
    )
    
    ypredZ = rep(NA, N)
    ypredC = rep(NA, N)
    
    ypredZ[Z_id] = stats::predict(fXZ, type = "response")
    ypredZ[C_id] = stats::predict(fXZ, newdata = dat[C_id,], type = "response")
    ypredZ[(!C_id) & (!Z_id)] = stats::predict(fXZ, newdata = dat[(!C_id) & (!Z_id),], type = "response")
    
    ypredC[Z_id] = stats::predict(fXC, newdata = dat[Z_id,], type = "response")
    ypredC[C_id] = stats::predict(fXC, type = "response")
    ypredC[(!C_id) & (!Z_id)] = stats::predict(fXC, newdata = dat[(!C_id) & (!Z_id),], type = "response")
    
    ## mean‑scale predictions on the full data set
    # ypredZ <- stats::predict(fXZ, newdata = dat, type = "response")
    # ypredC <- stats::predict(fXC, newdata = dat, type = "response")
    
  }
  
  ## optional “no‑harm” calibration
  if (calibration && (method != "lm")) {
    clb <- no_harm_clb(dat, ypredZ, ypredC, model_id)
    ypredZ <- clb$ypredZ_clb
    ypredC <- clb$ypredC_clb
  }
  
  # construct estimators
  tauhat = sum((dat$Y - ypredZ)*dat$Z*est_id)/sum(dat$Z*est_id) -
    sum((dat$Y - ypredC)*(1-dat$Z)*est_id)/sum((1-dat$Z)*est_id) +
    sum((ypredZ - ypredC)*est_id)/sum(est_id)
  
  varhat = sum((dat$Y - ypredZ)^2*dat$Z*est_id)/(sum(dat$Z*est_id)*(sum(dat$Z*est_id) - 1)) +
    sum((dat$Y - ypredC)^2*(1-dat$Z)*est_id)/(sum((1-dat$Z)*est_id)*(sum((1-dat$Z)*est_id) - 1))
  
  list(tauhat = tauhat, 
       varhat = varhat,
       fXZ = fXZ, 
       fXC = fXC)
}

## No harm calibration
no_harm_clb = function(dat, ypredZ, ypredC, model_id){
  N = nrow(dat)
  dat_run = dat
  dat_run$ypredZ = ypredZ # - mean(ypredZ)
  dat_run$ypredC = ypredC # - mean(ypredC)
  
  Z_id = (model_id == 1) & (dat_run$Z == 1)
  C_id = (model_id == 1) & (dat_run$Z == 0)
  
  fXZ_clb = lm(Y ~ ypredZ + ypredC, data = dat_run %>% filter(Z_id))
  fXC_clb = lm(Y ~ ypredZ + ypredC, data = dat_run %>% filter(C_id))
  
  ypredZ_clb = rep(NA, N)
  ypredC_clb = rep(NA, N)

  ypredZ_clb[Z_id] = stats::predict(fXZ_clb)
  ypredZ_clb[C_id] = stats::predict(fXZ_clb, newdata = dat_run[C_id,])
  ypredZ_clb[(!C_id) & (!Z_id)] = stats::predict(fXZ_clb, newdata = dat_run[(!C_id) & (!Z_id),])

  ypredC_clb[Z_id] = stats::predict(fXC_clb, newdata = dat_run[Z_id,])
  ypredC_clb[C_id] = stats::predict(fXC_clb)
  ypredC_clb[(!C_id) & (!Z_id)] = stats::predict(fXC_clb, newdata = dat_run[(!C_id) & (!Z_id),])
  
  # ypredZ_clb = predict(fXZ_clb, newdata = dat_run)
  # ypredC_clb = predict(fXC_clb, newdata = dat_run)
  
  list(ypredZ_clb = ypredZ_clb, 
       ypredC_clb = ypredC_clb)
}

## Cross fit
CF = function(dat, probZ, probS, method = "lm", calibration = FALSE, model_opts = list()){
  # Step 0: preparation
  N = nrow(dat)
  # S = rbinom(N, size = 1, probS)
  S = (seq(1, N) %% 2 == (1 - 1))
  dat_run = data.frame(
    dat, 
    S = S
  )
  
  # Step I: fit two separate estimators
  fit_q1 = part_fit(dat_run, model_id = S, est_id = (1-S), method, calibration, model_opts)
  fit_q2 = part_fit(dat_run, model_id = (1-S), est_id = S, method, calibration, model_opts)
  
  # Step II: obtain cross-fit estimators
  tau_CF = sum(1-S)/N * fit_q1$tauhat + sum(S)/N * fit_q2$tauhat
  var_CF = (sum(1-S)/N)^2 * fit_q1$varhat + (sum(S)/N)^2 * fit_q2$varhat
  
  list(tau_CF = tau_CF, var_CF = var_CF, tauhat = tau_CF, varhat = var_CF)
}

## DIM: Neyman's estimator (DIM)
DIM = function(dat){
  tau_DIM = mean(dat$Y[dat$Z == 1]) - mean(dat$Y[dat$Z == 0])
  var_DIM = var(dat$Y[dat$Z == 1])/sum(dat$Z == 1) + var(dat$Y[dat$Z == 0])/sum(dat$Z == 0)
  list(tau_DIM = tau_DIM, var_DIM = var_DIM, tauhat = tau_DIM, varhat = var_DIM)
}

## Fisher: Fisher's ANCOVA (ANCOVA)
ANCOVA = function(dat){
  # grab all names starting with "X"
  xvars = grep("^X", names(dat), value = TRUE)
  
  # (a) via paste/as.formula:
  form = as.formula(
    paste0("Y ~ Z + ", paste(xvars, collapse = " + "))
  )
  lm.fit = lm(form, data = dat)
  tau_ANCOVA = coef(lm.fit)["Z"]
  var_ANCOVA = hccm(lm.fit, type = "hc2")["Z", "Z"]
  list(tau_ANCOVA = tau_ANCOVA, var_ANCOVA = var_ANCOVA)
}

## classical: Lin's estimator (ADJ)
LIN = function(dat){
  # grab all names starting with "X"
  xvars = grep("^X", names(dat), value = TRUE)
  
  # (a) via paste/as.formula:
  form = as.formula(
    paste0("Y ~ ", paste(xvars, collapse = " + "))
  )
  
  # fit linear models
  N = nrow(dat)
  fit_LD = part_fit(dat, rep(1, N), rep(1, N))
  tau_LIN = fit_LD$tauhat
  var_LIN = hccm(fit_LD$fXZ, type = "hc3")[1, 1] + hccm(fit_LD$fXC, type = "hc3")[1, 1]
  list(tau_LIN = tau_LIN, 
       var_LIN = var_LIN)
}

## LD: Debiased estimator (LD)
LD = function(dat){
  # grab all names starting with "X"
  N = nrow(dat)
  xvars = grep("^X", names(dat), value = TRUE)
  X = as.matrix(dat[ , xvars])  
  XtX_inv = solve(t(X) %*% X)
  H = sapply(1:N, function(i){t(X[i,]) %*% XtX_inv %*% (X[i,]) })
  
  # run whole fit
  N1 = sum(dat$Z)
  N0 = sum(1-dat$Z)
  fit_LD = part_fit(dat, rep(1, N), rep(1, N))
  
  # obtain estimators and variance estimation based on HC3
  tau_LD = fit_LD$tauhat - 
    (sum(fit_LD$fXC$residuals * (diag(H)[dat$Z == 0]))/sum(1-dat$Z) * (N1/N0) - 
       sum(fit_LD$fXZ$residuals * (diag(H)[dat$Z == 1]))/sum(dat$Z) * (N0/N1))
  
  var_LD = hccm(fit_LD$fXZ, type = "hc3")[1, 1] + hccm(fit_LD$fXC, type = "hc3")[1, 1]
  
  # report results
  list(tau_LD = tau_LD, 
       var_LD = var_LD)
}


## FZ + linear regression (DC)
DC = function(dat, probZ, probRZ = NA, probRC = NA){
  # grab all names starting with "X"
  xvars = grep("^X", names(dat), value = TRUE)
  
  # Contsruct the formula
  form = as.formula(
    paste0("Y ~ ", paste(xvars, collapse = " + "))
  )
  
  # Specify sampling probabilities
  N = nrow(dat)
  p = length(xvars)
  
  if(is.na(probRZ)){
    probRZ = min(sqrt(p/N), 1/4)
  }
  if(is.na(probRC)){
    probRC = min(sqrt(p/N), 1/4)
  }
  
  probC = 1 - probZ 
  probMZ = (probZ - probRZ)/(1 - probRZ)
  probMC = (probC - probRC)/(1 - probRC)
  
  # draw Categorical variables
  DrawInd = sapply(
    dat$Z, 
    function(Z){
      ifelse(Z == 1, 
             sample.int(3, size = 1, replace = TRUE, prob = c(probMZ*probRZ/probZ, probMZ*(1 - probRZ)/probZ, (1 - probMZ)*probRZ) / probZ),
             sample.int(3, size = 1, replace = TRUE, prob = c(probMC*probRC/probC, probMC*(1 - probRC)/probC, (1 - probMC)*probRC) / probC))
    }
  )
  
  M = ifelse(DrawInd == 3, 0, 1)
  R = ifelse(DrawInd == 2, 0, 1)
  
  # Step I: fit regression models and construct DC estimators
  fitMC = part_fit(dat, R, M)
  
  # Step III: return the estimated value
  tau_DC = fitMC$tauhat
  var_DC = fitMC$varhat
  
  list(tau_DC = tau_DC, var_DC = var_DC)
}

## CF + linear regression (CF) for Bernoulli design
# CF = function(dat, probZ, probS){
#   # Step 0: preparation
#   N = nrow(dat)
#   S = rbinom(N, size = 1, probS)
#   dat_run = data.frame(
#     dat, 
#     S = S
#   )
#   
#   # Step I: fit two separate estimators
#   fit_q1 = part_fit(dat, model_id = S, est_id = (1-S))
#   fit_q2 = part_fit(dat, model_id = (1-S), est_id = S)
#   
#   # Step II: obtain cross-fit estimators
#   tau_CF = sum(S)/N * fit_q1$tauhat + sum(1-S)/N * fit_q2$tauhat
#   var_CF = (sum(S)/N)^2 * fit_q1$varhat + (sum(1-S)/N)^2 * fit_q2$varhat
#   
#   list(tau_CF = tau_CF, var_CF = var_CF)
# }


f1 = function(x)
{
  return(exp(-3 + 2*x)/(1+exp(-3 + 2*x)))
}
f = Vectorize(f1)
make_population_binom = function(N = 5e2, d = 1, probZ = .8){
  
  p = probZ
  mult = 1*(p-1)/p
  X = runif(N, -8, 8)
  pc= mult*(1.6)*(f(X)-1)
  pt= f(X)
  
  Y0 = rbinom(N, 1, pc)
  Y1 = rbinom(N, 1, pt)
  # Y0 = rpois(N, lambda = 3 + exp(X %*% beta0))
  
  tau = mean(Y1 - Y0)
  
  list(
    data = data.frame(ID = seq_len(N), X, Y1 = as.numeric(Y1), Y0 = Y0),
    tau  = tau
  )
}



# Based on code from swager/randomForest GitHub repo
ate.randomForest_calibrated = function(X, Y, W, nodesize = 20, conf.level=.9) {
  
  if (prod(W %in% c(0, 1)) != 1) {
    stop("Treatment assignment W must be encoded as 0-1 vector.")
  }
  
  N = nrow(as.matrix(X))
  pobs = ncol(X)
  
  if(length(unique(Y)) > 2) {
    # Sample-Split (two folds)
    tauVec = numeric(2)
    tauVec_cal = numeric(2)
    for(i in 1:2)
    {
      # Hold out half of the data points
      inds = (seq(1, N) %% 2 == (i - 1)) # select based on parity of the index 
      Y_leaveouti = Y[!inds]
      X_leaveouti = X[!inds,]
      W_leaveouti = W[!inds]
      
      initialFit0 = rep(NA, length(W_leaveouti))
      initialFit1 = rep(NA, length(W_leaveouti))
      
      yhat.0 = rep(NA, length(W_leaveouti))
      yhat.1 = rep(NA, length(W_leaveouti))
      
      rf_leaveouti.0 = randomForest::randomForest(X_leaveouti[W_leaveouti==0,], Y_leaveouti[W_leaveouti==0], nodesize = nodesize)
      rf_leaveouti.1 = randomForest::randomForest(X_leaveouti[W_leaveouti==1,], Y_leaveouti[W_leaveouti==1], nodesize = nodesize)
      
      # Form the initial predictions based upon the random forests (on the data points that aren't withheld)
      initialFit0[W_leaveouti==0] = stats::predict(rf_leaveouti.0)
      initialFit0[W_leaveouti==1] = stats::predict(rf_leaveouti.0, newdata = X_leaveouti[W_leaveouti==1,])
      initialFit1[W_leaveouti==1] = stats::predict(rf_leaveouti.1)
      initialFit1[W_leaveouti==0] = stats::predict(rf_leaveouti.1, newdata = X_leaveouti[W_leaveouti==0,])
      
      # Initial fit on the held out data
      tiHat = stats::predict(rf_leaveouti.1, newdata = X[inds,])
      ciHat = stats::predict(rf_leaveouti.0, newdata = X[inds,])
      
      # Form the calibrated predictions using the initial fits as covariates
      Xtilde = data.frame(initialFit1, initialFit0)
      Xtilde1 = Xtilde[W_leaveouti==1,]
      Xtilde0 = Xtilde[W_leaveouti==0,]
      lm1 = lm(Y_leaveouti[W_leaveouti==1]~., data = Xtilde1)
      lm0 = lm(Y_leaveouti[W_leaveouti==0]~., data = Xtilde0)
      calFit1 = predict(lm1, newdata = data.frame(initialFit1 = tiHat, initialFit0 = ciHat))
      calFit0 = predict(lm0, newdata = data.frame(initialFit1 = tiHat, initialFit0 = ciHat))
      
      n1 = sum(W[inds])
      n0 = sum(1 - W[inds])
      
      tauVec[i] = mean((tiHat - ciHat) + W[inds]*(N/n1)*(Y[inds] - tiHat) - (1 - W[inds])*(N/n0)*(Y[inds] - ciHat))
      tauVec_cal[i] = mean((calFit1 - calFit0) + W[inds]*(N/n1)*(Y[inds] - calFit1) - (1 - W[inds])*(N/n0)*(Y[inds] - calFit0))
    }
  } else {
    cat("Error: Not enough unique values of Y")
  }
  
  list(uncalibrated = mean(tauVec), calibrated = mean(tauVec_cal))
}

