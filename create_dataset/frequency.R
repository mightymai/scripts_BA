############## ADJUST HERE ##############
freq = 20
#########################################

# Einlesen der Datei
dat <- readRDS("data.rds")

# Entfernen von 'unnötigen' Spalten
dat$Prev <- NULL
dat$FBSFs <- NULL
dat$triphones <- NULL
dat$tetraphones <- NULL
dat$triphones <- NULL

head(dat)

frequency_overview <- table(dat$wordtoken)
head(frequency_overview)

dat_new <- frequency_overview[frequency_overview >= freq]
dat_new <- as.data.frame(dat_new)

result <- dat[dat$wordtoken %in% dat_new$Var1,]

result = result[order(result$wordtoken),]

write.csv(result,
          file = paste0('dat_', freq, '.csv'),
          row.names= FALSE,
          quote = FALSE)