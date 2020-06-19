suppressMessages(library(dplyr))
args = commandArgs((trailingOnly=TRUE))
#################################### predictVSword.R ####################################
#df_word
dat_word <- read.csv(args[1], sep = "\t", stringsAsFactors=FALSE)
df_word <- as.data.frame(dat_word)

running <- c("running_tap.wav","silence")
pink <- c("pink_noise.wav","silence")
white <- c("white_noise.wav","silence")
bike <- c("exercise_bike.wav","silence")
dude <- c("dude_miaowing.wav","silence")
dishes <- c("doing_the_dishes.wav","silence")
silence <- c("silence_0.wav","silence")
#df_word <- rbind(df_word, running, pink, white, bike, dude, dishes, silence)

df_word_sorted <- df_word[with(df_word, order(dat)),]


#df_predict
dat_predict <- read.csv(args[2], sep = ",", stringsAsFactors=FALSE)
df_predict <- as.data.frame(dat_predict)

df_predict_sorted <- df_predict[with(df_predict, order(fname)), ]

df_predict_sorted['word'] <- df_word_sorted[,2]

#rename colunm, more intuitive
names(df_predict_sorted)[names(df_predict_sorted) == "label"] <- "predict"
names(df_predict_sorted)[names(df_predict_sorted) == "fname"] <- "data"


#################################### confusion_matix.R ####################################

df <- df_predict_sorted
df <- df[!(df$predict=='silence'),]

################## table of confusion ##################
tmp <- df
tmp$correct <- df$predict == df$word
tmp$unknown <- df$predict == 'unknown'
tmp$FN <- (df$predict != df$word)

result <- aggregate(correct ~ word, tmp, FUN=sum)
result$unknown <- aggregate(unknown ~ word, tmp, FUN=sum)$unknown
result$FN <- aggregate(FN ~ word, tmp, FUN=sum)$FN
result$other <- result$FN - result$unknown
result$occurrence <- table(df$word)

test <- table(df$predict)
test <- as.data.frame(t(test))
test <- test[,2:3]
names(test)<-c("word","Freq")

result <- merge(result, test, by = 'word', all=TRUE)
result[is.na(result)] <-0

result$TP <- result$correct
result$FP <- result$Freq - result$correct
result <- result[c(1,2,3,5,6,7,8,9,4)]
result$TN <- nrow(df) - result$TP - result$FP - result$FN
rm(tmp)

for(i in 1:nrow(result)){
  sum_TP <- result[i,"TP"]
  sum_FP <- result[i,"FP"]
  sum_FN <- result[i,"FN"]
  sum_TN <- result[i,"TN"]
  
  wordtemp <- result[i,1]
  table_of_confusion <- data.frame(wordtemp = c("true", "false"),
                                   postive = c(sum_TP, sum_FN),
                                   negative= c(sum_FP, sum_TN))
  names(table_of_confusion) <- c(wordtemp, "positive", "negative")
}
print(noquote(paste0('number of tables of confusion created: ', nrow(result))))

################## zweite confusion matix ##################
val <- data.frame(matrix(0, nrow = nrow(result), ncol = nrow(result)))
val <- cbind(result$word, val)
names(val) <- c("word", t(result$word))
val <- subset(val, word != "unknown")
temp2 <- df

for(actual in 1:nrow(val)){
  for(pred in 2:(ncol(val) - 1)){
    temp_predict <- names(val)[pred] #aktuelle Spalte von predicted
    if(temp_predict %in% temp2$predict){ #prüft ob Wort überhaupt vorhergesagt wird. Wichtig für korrekte Dimension der table
      temp <- table(temp2$predict == temp_predict, temp2$word == val[actual,1])
      val[actual,pred] <- temp[2,2]
    }
  }
  print(noquote(paste0('current row ', actual, ' of ', nrow(result))))
}


df_new <- val[,2:ncol(val)]

df_new <- round(df_new/rowSums(df_new), 2)

output <- cbind(val[,1], df_new)
colnames(output) <- colnames(val)

output <- output%>%select(-unknown, unknown)


val <- val%>%select(-unknown, unknown)
#remove_silence <- c('silence')
#val[!(row.names(val) %in% remove_silence), ]
#cbind(val, val["silence"])
################## order matrix ##################

########### 1. accuracy ############
#data = pd.read_csv(table)    
#summe = 0 
#for rownum in range(len(data) - 1):
#  summe = summe + data.iloc[rownum][rownum + 1]
#print(str(summe) + ' und ' + str(rownum)) 
#
#acc = summe / (len(data) * 1) 
#acc = format(acc, '.4f')
#print('acc = ' + str(acc)) 
####################################

val$Summe <- rowSums(val[,-1])
val2 <- rbind(b = NA, val)
D <- diag(as.matrix(val2))
val$Diag <- D[-1]
val3 <- rbind(b = NA, output)
D2 <- diag(as.matrix(val3))
val$Diag_p <- D2[-1]

output$Summe <- val$Summe
#################################### scatter Plot ####################################
# 
# y_ax = val$Summe
# x_ax = val$Diag_p
# plot(y_ax, x_ax, ylab="Percentage of correct Predictions", xlab="Number of Occurence")
# abline(lm(x_ax ~ y_ax))

cf_name = args[3]
write.csv(val,
          file = paste0(cf_name, "_val.csv"),
          row.names= FALSE,
          quote = FALSE)

write.csv(output,
          file = paste0(cf_name, ".csv"),
          row.names= FALSE,
          quote = FALSE)