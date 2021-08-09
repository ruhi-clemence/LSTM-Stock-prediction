library(tibble)
library(readr)
library(ggplot2)
library(timeSeries)
library(zoo)
library(xts)
library(timeDate)
library(timetk)
library(timeSeries)
library(keras)
library(tensorflow)
library(Metrics)

head(Safaricom)
tail(Safaricom) 
price<-Safaricom$Price
date<-Safaricom$Date
Safaricom.TS = xts(price,order.by=date)
Safaricom.ts = tk_ts(Safaricom.TS)
Date = tk_index(Safaricom.ts, timetk_idx = FALSE)
head(Date)
adj_price = c(mean(price), sd(price))
new_price = (price - adj_price[1]) / adj_price[2]
head(new_price)
tail(new_price)
Safaricom.ts<- xts(new_price, order.by = date)
summary(Safaricom.ts)
plot(Safaricom.ts)
diffed =na.omit(diff(Safaricom.ts, difference = 1))
head(diffed)

#create a lagged dataset to be supervised
lags <- function(x, k=1){ 
  lagged =  c(rep(NA, k), x[1:(length(x)-k)]) 
  DF = as.data.frame(cbind(lagged, x)) 
  colnames(DF) <- c( paste0('x-', k), 'x') 
  DF[is.na(DF)] <- 0 
  return(DF) 
} 
supervised = lags(Safaricom.ts, 1) 
head(supervised)

#split first 70% of the data as training and the rest 30% as test sets
N = nrow(supervised) 
n = round(N *0.7, digits = 0) 
train = supervised[1:n, ] 
test  = supervised[(n+1):N,  ] 

## scale data 
scale_data <- function(train, test, feature_range = c(0, 1)) { 
  x = train 
  fr_min = feature_range[1] 
  fr_max = feature_range[2] 
  std_train = ((x - min(x) ) / (max(x) - min(x)  )) 
  std_test  = ((test - min(x) ) / (max(x) - min(x)  )) 
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min 
  scaled_test = std_test *(fr_max -fr_min) + fr_min 
  
  return( list(scaled_train = as.vector(scaled_train), 
               scaled_test = as.vector(scaled_test) ,
               scaler= c(min =min(x), max = max(x))) ) 
} 

#revert the predicted values to the original scale
invert_scaling = function(scaled, scaler, feature_range = c(0, 1)){ 
  min = scaler[1] 
  max = scaler[2] 
  t = length(scaled) 
  mins = feature_range[1] 
  maxs = feature_range[2] 
  inverted_dfs = numeric(t) 
  
  for( i in 1:t){ 
    X = (scaled[i]- mins)/(maxs - mins) 
    rawValues = X *(max - min) + min 
    inverted_dfs[i] <- rawValues 
  } 
  return(inverted_dfs) 
} 



scaled = scale_data(train, test, c(-1, 1))
y_train = scaled$scaled_train[, 2]
x_train = scaled$scaled_train[, 1]
y_test = scaled$scaled_test[, 2]
x_test = scaled$scaled_test[, 1]

# fit the model 
dim(x_train) <- c(length(x_train), 1, 1) 
dim(x_train) 
X_shape2 = dim(x_train)[2] 
X_shape3 = dim(x_train)[3] 
batch_size = 1
units = 1 
model <- keras_model_sequential()  
model%>% 
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3),
             stateful= TRUE)%>% 
  layer_dense(units = 1) 
model %>% compile( 
  loss = 'mean_squared_error', 
  optimizer = 'adam',   
  metrics = c('accuracy') 
) 

summary(model) 

#Fit the model
EpochS = 20
for(i in 1:EpochS ){ 
  model %>% fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=1, 
                shuffle=FALSE) 
  model %>% reset_states() 
}

#make price predictions
L = length(x_test) 
dim(x_test) = c(length(x_test), 1, 1) 
scaler = scaled$scaler ;scaler
prediction = numeric(L)
for(i in 1:L){
  X = x_test[i, 1, 1] 
  dim(X) = c(1,1,1) 
yhat = model %>% predict(X, batch_size=batch_size) 
yhat = invert_scaling(yhat, scaler,  c(-1, 1)) 
yhat  = yhat + Safaricom.TS[(n+i)]  
prediction[i] <- yhat 
}
tail(prediction)
                   

L = length(x_test) 
dim(x_test) = c(length(x_test), 1, 1) 
scaler = scaled$scaler ;scaler
predictions = numeric(L)
for(i in 1:L){
  X = x_test[i, 1, 1] 
  dim(X) = c(1,1,1) 
  yhat = model %>% predict(X, batch_size=batch_size) 
  yhat = invert_scaling(yhat, scaler,  c(-1, 1)) 
  yhat  = yhat + Safaricom.ts[(n+i)]  
  predictions[i] <- yhat 
}
head(predictions) 



result = rmse(test$x, predictions)
results = mae(test$x, predictions)