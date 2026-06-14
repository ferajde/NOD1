# ====================================================================
# Laboratorium 5 — Regresja: porównanie modeli w R
# Dataset: mtcars (wbudowany w R), target: mpg (miles per gallon)
# ====================================================================
# Pakiety: tylko bazowe + MASS (Ridge) + nnet (sieć neuronowa)
# Oba MASS i nnet są częścią standardowej instalacji R — nie wymagają install.packages()
# ====================================================================

library(MASS)
library(nnet)
set.seed(42)

# ====================================================================
# 1. ZAŁADOWANIE DANYCH
# ====================================================================
cat("=== 1. DANE ===\n")
data(mtcars)
cat("Wymiary mtcars:", dim(mtcars)[1], "x", dim(mtcars)[2], "\n")
cat("Kolumny:", colnames(mtcars), "\n")
cat("Target: mpg (zużycie paliwa)\n\n")

# Podział train/test 80/20
n <- nrow(mtcars)
idx <- sample(seq_len(n), size = floor(0.8 * n))
train <- mtcars[idx, ]
test  <- mtcars[-idx, ]
cat("Train:", nrow(train), "Test:", nrow(test), "\n\n")

# ====================================================================
# 2. MODELE KLASYCZNE: LINEAR I RIDGE
# ====================================================================
cat("=== 2. MODELE KLASYCZNE ===\n\n")

# --- Regresja liniowa ---
model_lm <- lm(mpg ~ ., data = train)
y_pred_lm <- predict(model_lm, newdata = test)
y_test <- test$mpg

r2_lm  <- 1 - sum((y_test - y_pred_lm)^2) / sum((y_test - mean(y_test))^2)
mse_lm <- mean((y_test - y_pred_lm)^2)
mae_lm <- mean(abs(y_test - y_pred_lm))

cat("Linear:  R² =", round(r2_lm, 3),
    " MSE =", round(mse_lm, 2),
    " MAE =", round(mae_lm, 2), "\n")

# --- Ridge (MASS::lm.ridge) ---
# lm.ridge wymaga jawnej macierzy cech
X_train <- as.matrix(train[, -1])  # bez mpg
y_train <- train$mpg
X_test  <- as.matrix(test[, -1])

# Dopasowanie Ridge dla różnych lambda
lambdas <- seq(0, 20, 0.5)
ridge_models <- lm.ridge(mpg ~ ., data = train, lambda = lambdas)

# Wybór najlepszego lambda wg GCV (Generalized Cross-Validation)
best_lambda <- lambdas[which.min(ridge_models$GCV)]
cat("Najlepsze lambda Ridge (wg GCV):", best_lambda, "\n")

# Predykcja Ridge — ręcznie z współczynników
ridge_best <- lm.ridge(mpg ~ ., data = train, lambda = best_lambda)
coefs <- coef(ridge_best)
y_pred_ridge <- coefs[1] + X_test %*% coefs[-1]
y_pred_ridge <- as.vector(y_pred_ridge)

r2_ridge  <- 1 - sum((y_test - y_pred_ridge)^2) / sum((y_test - mean(y_test))^2)
mse_ridge <- mean((y_test - y_pred_ridge)^2)
mae_ridge <- mean(abs(y_test - y_pred_ridge))

cat("Ridge (lambda=", best_lambda, "):  R² =", round(r2_ridge, 3),
    " MSE =", round(mse_ridge, 2),
    " MAE =", round(mae_ridge, 2), "\n\n")

# ====================================================================
# 3. SIEĆ NEURONOWA (nnet)
# ====================================================================
cat("=== 3. SIEĆ NEURONOWA ===\n\n")

# nnet wymaga znormalizowanych danych dla stabilności
sx <- scale(X_train)
sx_test <- scale(X_test, center = attr(sx, "scaled:center"),
                 scale = attr(sx, "scaled:scale"))
sy_train <- scale(y_train)
y_mean <- attr(sy_train, "scaled:center")
y_sd   <- attr(sy_train, "scaled:scale")

# Sieć: 10 cech → 5 ukrytych neuronów → 1 wyjście (linout=TRUE dla regresji)
set.seed(42)
nn_model <- nnet(sx, sy_train,
                 size = 5, linout = TRUE,
                 decay = 0.01, maxit = 500, trace = FALSE)

# Predykcja w skali ORYGINALNEJ
y_pred_nn_s <- predict(nn_model, sx_test)
y_pred_nn <- y_pred_nn_s * y_sd + y_mean
y_pred_nn <- as.vector(y_pred_nn)

r2_nn  <- 1 - sum((y_test - y_pred_nn)^2) / sum((y_test - mean(y_test))^2)
mse_nn <- mean((y_test - y_pred_nn)^2)
mae_nn <- mean(abs(y_test - y_pred_nn))

cat("Sieć neuronowa (5 neuronów ukr.):  R² =", round(r2_nn, 3),
    " MSE =", round(mse_nn, 2),
    " MAE =", round(mae_nn, 2), "\n\n")

# ====================================================================
# 4. PORÓWNANIE MODELI — TABELA
# ====================================================================
cat("=== 4. PORÓWNANIE MODELI ===\n")
porownanie <- data.frame(
  Model = c("Linear", paste0("Ridge (lambda=", best_lambda, ")"), "NN (5 ukr.)"),
  R2  = round(c(r2_lm, r2_ridge, r2_nn), 3),
  MSE = round(c(mse_lm, mse_ridge, mse_nn), 2),
  MAE = round(c(mae_lm, mae_ridge, mae_nn), 2)
)
print(porownanie)
cat("\n")

# ====================================================================
# 5. WAŻNOŚĆ CECH W RIDGE
# ====================================================================
cat("=== 5. WAŻNOŚĆ CECH (Ridge) ===\n")
importance <- abs(coefs[-1])  # bez wyrazu wolnego
names(importance) <- colnames(X_train)
importance_sorted <- sort(importance, decreasing = TRUE)
print(round(importance_sorted, 4))
cat("\n")

# ====================================================================
# 6. ANALIZA RESZT — REGRESJA LINIOWA
# ====================================================================
cat("=== 6. ANALIZA RESZT (model liniowy) ===\n")
residuals_lm <- y_test - y_pred_lm

cat("Średnia reszt:", round(mean(residuals_lm), 4), "(powinna być ~0)\n")
cat("Odch. std reszt:", round(sd(residuals_lm), 4), "\n\n")

# Test normalności Shapiro-Wilka
sw <- shapiro.test(residuals_lm)
cat("Test Shapiro-Wilka:  W =", round(sw$statistic, 4),
    " p-value =", round(sw$p.value, 4), "\n")
if (sw$p.value > 0.05) {
  cat("  -> p > 0.05: brak podstaw do odrzucenia H0, reszty zgodne z N(0, sigma^2)  OK\n")
} else {
  cat("  -> p < 0.05: odrzucamy H0, reszty NIE sa w pelni normalne\n")
}
cat("\n")

# Durbin-Watson (ręcznie — bez pakietu lmtest)
# DW = sum((e_t - e_{t-1})^2) / sum(e_t^2)
dw_manual <- function(r) sum(diff(r)^2) / sum(r^2)
dw <- dw_manual(residuals_lm)
cat("Statystyka Durbin-Watsona: DW =", round(dw, 4), "\n")
cat("  DW ~ 2 -> brak autokorelacji  (idealnie)\n")
cat("  DW < 1.5 -> autokorelacja dodatnia\n")
cat("  DW > 2.5 -> autokorelacja ujemna\n")
if (dw > 1.5 && dw < 2.5) {
  cat("  -> DW =", round(dw, 2), " w przedziale (1.5, 2.5): brak istotnej autokorelacji  OK\n")
} else {
  cat("  -> DW =", round(dw, 2), " sygnalizuje autokorelacje reszt\n")
}
cat("\n")

# ====================================================================
# 7. SKALA: ORYGINALNE vs ZNORMALIZOWANE
# ====================================================================
cat("=== 7. SKALA DANYCH: ORYGINALNE vs ZNORMALIZOWANE ===\n")

# Sztucznie "rozjedzmy" skale jednej cechy x1000
train_orig <- train; test_orig <- test
train_orig$cyl <- train_orig$cyl * 1000
test_orig$cyl  <- test_orig$cyl * 1000

# Wersja znormalizowana
train_norm <- train_orig
test_norm  <- test_orig
mu <- colMeans(train_orig); sd_vec <- apply(train_orig, 2, sd)
for (col in colnames(train_orig)[-1]) {  # bez mpg
  train_norm[[col]] <- (train_orig[[col]] - mu[col]) / sd_vec[col]
  test_norm[[col]]  <- (test_orig[[col]]  - mu[col]) / sd_vec[col]
}

cmp_skala <- function(label, dtr, dte) {
  m <- lm(mpg ~ ., data = dtr)
  p <- predict(m, newdata = dte)
  r2 <- 1 - sum((dte$mpg - p)^2) / sum((dte$mpg - mean(dte$mpg))^2)
  mae <- mean(abs(dte$mpg - p))
  cat(sprintf("  %-30s R^2 = %.3f  MAE = %.3f\n", label, r2, mae))
}

cat("Regresja liniowa lm():\n")
cmp_skala("oryginalne (zdeformowane)", train_orig, test_orig)
cmp_skala("znormalizowane", train_norm, test_norm)

# Dla Ridge — używamy lm.ridge i lambda=1
ridge_cmp <- function(label, dtr, dte, lambda = 1) {
  m <- lm.ridge(mpg ~ ., data = dtr, lambda = lambda)
  cf <- coef(m)
  X_te <- as.matrix(dte[, -1])
  p <- cf[1] + X_te %*% cf[-1]; p <- as.vector(p)
  r2  <- 1 - sum((dte$mpg - p)^2) / sum((dte$mpg - mean(dte$mpg))^2)
  mae <- mean(abs(dte$mpg - p))
  cat(sprintf("  %-30s R^2 = %.3f  MAE = %.3f\n", label, r2, mae))
}

cat("Ridge (lambda=1):\n")
ridge_cmp("oryginalne (zdeformowane)", train_orig, test_orig)
ridge_cmp("znormalizowane", train_norm, test_norm)

cat("\nUwaga: w R funkcja MASS::lm.ridge() DOMYSLNIE standaryzuje cechy wewnetrznie,\n")
cat("dlatego R^2 dla oryginalnych i znormalizowanych jest identyczne.\n")
cat("W sklearn.Ridge tej standaryzacji nie ma -> tam roznica byla wyrazna.\n")
cat("Wniosek: lm() z definicji odporna na skale; Ridge w R bezpieczna domyslnie,\n")
cat("ale w innych ekosystemach (sklearn, glmnet z standardize=FALSE) trzeba pamietac.\n\n")

# ====================================================================
# 8. WYKRESY -> PDF
# ====================================================================
cat("=== 8. WYKRESY ===\n")
pdf("zad_5_R_wykresy.pdf", width = 9, height = 6)

# 8.1 Porównanie metryk
par(mfrow = c(1, 2), mar = c(7, 4, 3, 1))
barplot(porownanie$R2, names.arg = porownanie$Model,
        col = c("steelblue","darkgreen","tomato"), las = 2,
        main = "R^2 (wieksze = lepsze)", ylab = "R^2")
barplot(porownanie$MAE, names.arg = porownanie$Model,
        col = c("steelblue","darkgreen","tomato"), las = 2,
        main = "MAE (mniejsze = lepsze)", ylab = "MAE")

# 8.2 Ważność cech Ridge
par(mfrow = c(1, 1), mar = c(7, 4, 3, 1))
barplot(importance_sorted, las = 2,
        col = "steelblue", main = "Waznosc cech (|beta| w modelu Ridge)",
        ylab = "|wspolczynnik|")

# 8.3 Histogram reszt
par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
hist(residuals_lm, breaks = 12, col = "lightblue",
     main = "Histogram reszt (Linear)", xlab = "Reszty")
abline(v = 0, col = "red", lwd = 2)

# 8.4 Q-Q plot reszt
qqnorm(residuals_lm, main = "Q-Q Plot reszt", pch = 19)
qqline(residuals_lm, col = "red", lwd = 2)

# 8.5 Predykcja vs Rzeczywistość — wszystkie modele
par(mfrow = c(1, 1), mar = c(4, 4, 3, 1))
plot(y_test, y_pred_lm, pch = 19, col = "steelblue",
     xlab = "Wartosc rzeczywista (mpg)",
     ylab = "Predykcja",
     main = "Predykcja vs Rzeczywistosc",
     xlim = range(c(y_test, y_pred_lm, y_pred_ridge, y_pred_nn)),
     ylim = range(c(y_test, y_pred_lm, y_pred_ridge, y_pred_nn)))
points(y_test, y_pred_ridge, pch = 19, col = "darkgreen")
points(y_test, y_pred_nn,    pch = 19, col = "tomato")
abline(0, 1, lty = 2, col = "black")
legend("topleft", legend = c("Linear","Ridge","NN","y=x"),
       col = c("steelblue","darkgreen","tomato","black"),
       pch = c(19,19,19,NA), lty = c(NA,NA,NA,2))

dev.off()
cat("Wykresy zapisane do: zad_5_R_wykresy.pdf\n\n")

# ====================================================================
# PODSUMOWANIE
# ====================================================================
cat("===================================================\n")
cat("PODSUMOWANIE\n")
cat("===================================================\n")
cat("Dataset:       mtcars (32 obs., 11 zmiennych), target: mpg\n")
cat("Najlepszy model:", as.character(porownanie$Model[which.max(porownanie$R2)]), "\n")
cat("Pakiety:       MASS (Ridge) + nnet (NN) — oba w bazowym R, bez instalacji\n")
cat("Najsilniejsza cecha (Ridge):", names(importance_sorted)[1],
    "(|beta| =", round(importance_sorted[1], 3), ")\n")
cat("Shapiro p-value:", round(sw$p.value, 4), "\n")
cat("Durbin-Watson:  ", round(dw, 4), "\n")
cat("===================================================\n")
