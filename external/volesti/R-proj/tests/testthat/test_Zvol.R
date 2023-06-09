context("Zonotopes' volume test")

library(volesti)

Zruntest <- function(P, name_string, tol, num_of_exps, algo, seed){
  
  exactvol = exact_vol(P)
  vol = 0
  for (j in 1:num_of_exps) {
    if (algo == "CB") {
      vol = vol + volume(P, settings = list("hpoly" = FALSE), rounding = FALSE, seed = seed)
    } else {
      vol = vol + volume(P, settings = list("algorithm" = "CG", "error" = 0.1), rounding = FALSE, seed = seed)
    }
  }
  vol = vol / num_of_exps
  error = abs(vol - exactvol) / exactvol
  if (error >= tol){
    res = 0
  } else {
    res = 1
  }
  return(res)
}

cran_only = TRUE
num_of_exps = 2

for (i in 1:2) {
  if (i==1) {
    algo = 'CG'
    tol = 0.2
  } else {
    algo = 'CB'
    tol = 0.2
  }

  test_that("Volume Zonotope_2_4", {
    #skip_if(Sys.info()[["machine"]] %in% c("x86_32"))
    Z = gen_rand_zonotope(2, 4, seed = 127)
    res = Zruntest(Z, 'Zonotope_2_4', tol, num_of_exps, algo, 5)
    expect_equal(res, 1)
  })
  
}
