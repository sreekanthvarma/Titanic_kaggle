library(ggplot2)
library(dplyr)

train <- read.csv("Titanic_train.csv")
test <- read.csv("Titanic_test.csv")

### Data Analysis

## Missing values

temp <- train
temp[temp == ""] <- NA

## Number of missing values per column 
colSums(is.na(temp))

## Percent of missing values per column
round(colSums(is.na(temp))/nrow(temp), 3)

## We can see that 77% of the values in cabin are missing. So it is not safe to include this data to the model.
## 2 values are missing for embarked which we can estimate by using mode
## Age has about 20% of values missing. We can either replace the missing values with mean or we can use the mice package to
## impute those missing values

## Analyzing by columns
## Pclass

# Total pasengers per Pclass
ggplot(data = train, aes(x = Pclass)) +
  geom_bar() 

# Passengers survived per Pclass
ggplot(data = train, aes(x = Pclass, y = Survived, fill = Pclass)) +
  geom_bar(stat = "identity")

# Survival rate per Pclass
survival_class <- train %>%
  group_by(Pclass) %>%
  summarize(total_passengers = n(), survived = sum(Survived), survival_rate = mean(Survived))

survival_class

# Pclass 1 has higher survival rate compared to Pclass 2 which has higher than Pclass 3. So Pclass could be a significant factor in
# determining the survival rate


## Sex
ggplot(data = train, aes(x = Sex)) +
  geom_bar() 

# Passengers survived per Pclass
ggplot(data = train, aes(x = Sex, y = Survived, fill = Sex)) +
  geom_bar(stat = "identity")

# Survival rate per Pclass
survival_sex <- train %>%
  group_by(Sex) %>%
  summarize(total_passengers = n(), survived = sum(Survived), survival_rate = mean(Survived))

survival_sex

## Females have a very high survival rate of 74.2% compared to males who have a survival rate of just 18.9%


## Ticket
length(unique(train$Ticket))

length(unique(train$Ticket))/nrow(train)
## As expected we have a lot of unique values for ticket which are 681 unique values accounting for 76.4% of values. 
## Since ticket is more unique, it doesnt impact the prediction much, so we can ignore this column


## Sib and Parch

ggplot(data = train, aes(x = SibSp)) +
  geom_bar() 

# Passengers survived per Pclass
ggplot(data = train, aes(x = SibSp, y = Survived, fill = SibSp)) +
  geom_bar(stat = "identity")

# Survival rate per Pclass
survival_sib <- train %>%
  group_by(SibSp) %>%
  summarize(total_passengers = n(), survived = sum(Survived), survival_rate = mean(Survived))

survival_sib

## People with one or two siblings/spouses have more chance of survival compared to other numbers


## Parch
ggplot(data = train, aes(x = Parch)) +
  geom_bar() 

# Passengers survived per Pclass
ggplot(data = train, aes(x = Parch, y = Survived, fill = Parch)) +
  geom_bar(stat = "identity")

# Survival rate per Pclass
survival_parch <- train %>%
  group_by(Parch) %>%
  summarize(total_passengers = n(), survived = sum(Survived), survival_rate = mean(Survived))

survival_parch

## Families of size 2-4 have more chances of survival compared to people who were travelling alone.


## Embarked
ggplot(data = train, aes(x = Embarked)) +
  geom_bar() 

# Passengers survived per Pclass
ggplot(data = train, aes(x = Embarked, y = Survived, fill = Embarked)) +
  geom_bar(stat = "identity")

# Survival rate per Pclass
survival_emb <- train %>%
  group_by(Embarked) %>%
  summarize(total_passengers = n(), survived = sum(Survived), survival_rate = mean(Survived))

survival_emb
## We can notice that people with port of embarkation of Cherbourg had relatively high chance of survival compared to other
## two ports. So even port of embarkation might have some effect on survival rate


## Age
## Since we are missing close to 20% of values of age, we need to impute them before making any predictions
library(mice)

target <- train$Survived
train$Survived <- NULL

## To apply feature engineering and fix few columns, I am going to merge both the train and test datasets and then divide them
## back later

toimpute <- rbind(train, test)
toimpute$Cabin <- NULL
toimpute$PassengerId <- NULL
toimpute$Name <- NULL
toimpute$Ticket <- NULL

imputed <- mice(toimpute, m=5, maxit=500, seed=99,nnet.MaxNWts = 2600)
titanic_complete <- complete(imputed,1)


## Age and Fare are continuous variables and standardizing them should give a better response

## Prediction
library(caret)

## Using log transformation to standardize the age and fare variables
titanic_complete$Age <- log(titanic_complete$Age+1)
titanic_complete$Fare <- log(titanic_complete$Fare+1)
titanic_complete$Embarked[titanic_complete$Embarked == ''] <- "S"

## Converting Pclass, sex and embarked to factors
titanic_complete <- titanic_complete %>%
  mutate(Pclass = as.factor(Pclass),
         Sex = as.character(Sex),
         Embarked = as.character(Embarked)) %>%
  mutate(Sex = ifelse(Sex == "male", 1, 0),
         Embarked = ifelse(Embarked == "C", 1, ifelse(Embarked == "Q", 2, 3))) %>%
  mutate(Sex = as.factor(Sex),
         Embarked = as.factor(Embarked))

glimpse(titanic_complete)


## Dividing the merged train-test dataset back to separate datasets
train_model <- titanic_complete[1:nrow(train),]
test_model <- titanic_complete[(nrow(train)+1):nrow(titanic_complete), ]
target_model <- data.frame(as.factor(target))

## Separating train data to a new train and validation datasets so that we can check our model accuracy on the validation set and
## can tune our model accordingly

## Dividing train by a split ratio of 70:30
smp_size <- floor(0.7 * nrow(train_model))

set.seed(777)
ind <- sample(seq_len(nrow(train_model)), size = smp_size)

train_1 <- train_model[ind, ]
validation_1 <- train_model[-ind, ]

target_1 <- target_model[ind, ]
target_val_1<- target_model[-ind, ]

## Training the Naive Bayes model based on train dataset
model_nb <- train(train_1, target_1, 'nb', trControl=trainControl(method = 'cv', number = 10))

prediction_nb_1 <- predict(model_nb, validation_1)

xtab_nb <- table(prediction_nb_1, target_val_1)                      

accuracy_nb <- (xtab[1,1]+xtab[2,2])/sum(xtab)
## The model gave an accuracy of 75.3% on the validation dataset

library(pROC)
validation_target <- as.numeric(validation_target)
auc(prediction_val, validation_target)
## Area under the curve value came out to be 0.7378 on the validation dataset


#### NB model output
prediction_nb <- data.frame(Survived =predict(model_nb, test_model))



## XGBoost
library(xgboost)

Pclass <- titanic_complete %>%
  with(model.matrix(~Pclass + 0))

Sex <- titanic_complete %>%
  with(model.matrix(~Sex + 0))

Embarked <- titanic_complete %>%
  with(model.matrix(~Embarked + 0))

titanic_xgb <- cbind(data.frame(Pclass), data.frame(Sex), data.frame(Embarked), titanic_complete)
titanic_xgb[c("Pclass", "Sex", "Embarked")] <- NULL

train_xgb <- as.matrix(titanic_xgb[1:nrow(train),])
test_xgb <- as.matrix(titanic_xgb[(nrow(train)+1):nrow(titanic_xgb), ])
target_xgb <- as.matrix(data.frame(as.factor(target)))


target_xgb <- ifelse(target_model$as.factor.target. == 1, 1, 0)


## Model tuning by breaking train into train and validation
train_xgb_1 <- train_xgb[ind, ]
val_xgb_1 <- train_xgb[-ind, ]

target_xgb_1 <- target_xgb[ind, ]
target_val_1 <- target_xgb[-ind, ]

# rounds <- seq(100, 1000, 100)
# depth <- seq(2,7)
# accuracy <- matrix(NA, nrow = length(rounds), ncol = length(depth))
# 
# for (i in 1:length(rounds)){
#   for (j in 1:length(depth)){
#     model_xgb <- xgboost(data = train_xgb_1, label = target_xgb_1, max_depth = depth[j],
#                          eta = 0.1, print_every_n = 5, nthread = 4, nrounds = rounds[i],
#                          objective = "binary:logistic")
#     
#     pred_xgb <- data.frame(prediction = predict(model_xgb, val_xgb_1, type = "class"))
#     pred_xgb <- ifelse(pred_xgb$prediction > 0.5, 1, 0)
#     xtab_xgb <- table(data.frame(pred_xgb, target_val_1))
#     accuracy[i,j] <- (xtab_xgb[1,1]+xtab_xgb[2,2])/sum(xtab_xgb)
#   }
# }

## Model based on tuned parameters
model_xgb <- xgboost(data = train_xgb, label = target_xgb, max_depth = 4,
                     eta = 0.1, print_every_n = 5, nthread = 4, nrounds = 100,
                     objective = "binary:logistic")

## Making prediction on the test data using the above tuned model
pred_xgb <- data.frame(Survived = predict(model_xgb, test_xgb, type = "class"))
pred_xgb <- ifelse(pred_xgb$Survived > 0.5, 1, 0)


output_xgb <- data.frame(PassengerId = seq(892, 1309), Survived = pred_xgb)
write.csv(output_xgb, "Titanic_R_xgb.csv", row.names = FALSE)



## RandomForest
library(randomForest)
colnames(target_model) <- c("target")
train_rf <- data.frame(target_model, train_model)
test_rf <- test_model
train_rf$target <- as.factor(target)
model_rf <- randomForest(target ~ ., data = train_model, ntree = 300, mtry = 4)
pred_rf <- data.frame(Survived = predict(model_rf, test_model))

pred_rf <- ifelse(pred_rf$Survived > 0.6, 1, 0)

output_rf <- data.frame(PassengerId = seq(892, 1309), Survived = pred_rf)


## Mode of 3 models
final <- data.frame(output_xgb$Survived, output_rf$Survived, prediction_nb)
library(modeest)
output <- data.frame(PassengerId = seq(892, 1309), Survived = NA)
output$Survived <- apply(final, 1, mfv)

## Assigning weights to models
final <- data.frame(output_xgb$Survived*3, output_rf$Survived*2, prediction_nb)
final$Survived <- ifelse(prediction_nb == 1, 1, 0)

output <- data.frame(PassengerId = seq(892, 1309), Survived = NA)
output$Survived <- rowSums(final)/6
output$Survived <- ifelse(output$Survived >= 0.5, 1, 0)

## Saving out the file with passenger id and survived as two categories
write.csv(output, "Titanic_R_ensemble_2.csv", row.names = FALSE)

