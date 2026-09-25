#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(limma))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("usage: fit_limma_moderation.R plate_relative_signatures.csv.gz output_dir")
}

input_path <- args[[1]]
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

development_ids <- c(
  "BRD-K81418486",
  "BRD-A19037878",
  "BRD-A75409952",
  "BRD-A19500257"
)

frame <- read.csv(input_path, check.names = FALSE, stringsAsFactors = FALSE)
frame <- frame[frame$pert_id %in% development_ids, ]
if (length(unique(frame$pert_id)) != length(development_ids)) {
  stop("one or more frozen development compounds are missing")
}

gene_columns <- grep("^g_", colnames(frame), value = TRUE)
expression <- t(as.matrix(frame[, gene_columns]))
storage.mode(expression) <- "double"
group <- factor(frame$pert_id, levels = development_ids)
design <- model.matrix(~ 0 + group)
colnames(design) <- development_ids

fit <- lmFit(expression, design)
fit <- eBayes(fit, trend = TRUE, robust = FALSE)

moderation <- data.frame(
  gene_id = sub("^g_", "", gene_columns),
  residual_variance = fit$sigma^2,
  posterior_variance = fit$s2.post,
  precision_weight = 1 / fit$s2.post,
  prior_df = fit$df.prior,
  residual_df = fit$df.residual
)

coefficients <- data.frame(gene_id = moderation$gene_id, fit$coefficients, check.names = FALSE)
write.csv(moderation, file.path(output_dir, "limma_gene_moderation.csv"), row.names = FALSE)
write.csv(coefficients, file.path(output_dir, "development_compound_effects.csv"), row.names = FALSE)

qa <- list(
  limma_version = as.character(packageVersion("limma")),
  n_samples = ncol(expression),
  n_genes = nrow(expression),
  n_compounds = length(unique(group)),
  design_rank = qr(design)$rank,
  median_prior_df = median(fit$df.prior),
  median_residual_df = median(fit$df.residual),
  all_posterior_variances_finite_positive = all(is.finite(fit$s2.post) & fit$s2.post > 0)
)
dput(qa, file = file.path(output_dir, "limma_fit_qa.dput"))
print(qa)
