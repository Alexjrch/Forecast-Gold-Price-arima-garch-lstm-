%% ARIMA-GARCH
diffprice = diff(price);
figure;
subplot(2,1,1);
autocorr(diffprice);
subplot(2,1,2)
parcorr(diffprice);

%%
arimatrain = diffprice(1:end);
VarMdl = garch(1,1);
Mdl = arima('MALags',1,'ARLags',1,'Constant',0,'Variance',VarMdl);
EstMdl = estimate(Mdl,arimatrain);
summarize(EstMdl);
[Y,YMSE] = forecast(EstMdl,5,diffprice);
[res,~,logL] = infer(EstMdl,diffprice);
%%
arimatrain = diffprice(1:end);
VarMdl = garch(1,1);
Mdl = arima('ARLags',1,'MALags',1,'Constant',0,'Variance',VarMdl);
EstMdl = estimate(Mdl,arimatrain);
summarize(EstMdl);
[Y,YMSE] = forecast(EstMdl,5,diffprice);
[res,~,logL] = infer(EstMdl,diffprice);

figure;
subplot(2,2,1);
plot(res);
title('Residuals');
subplot(2,2,2);
histogram(res,50);
title('Residuals');
subplot(2,2,3);
autocorr(res);
subplot(2,2,4);
parcorr(res);

%%
forecastprice = [price(end);0;0;0;0;0];
for i = 2:length(Y)
    forecastprice(i) = forecastprice(i-1) + Y(i-1);
    forecastprice(i)
    Y(i-1)
end

%%
lower = Y - 1.96*sqrt(YMSE);
upper = Y + 1.96*sqrt(YMSE);
figure
plot(diffprice,'Color',[.7,.7,.7]);
hold on
h1 = plot(2087:2608,lower,'r:','LineWidth',2);
plot(2087:2608,upper,'r:','LineWidth',2)
h2 = plot(2087:2608,Y,'k','LineWidth',2);
legend([h1 h2],'95% Interval','Forecast',...
	     'Location','NorthWest')
title('Gold Price Forecast')
hold off

%% LSTM
gprice = price';
%%
numTimeStepsTrain = floor(0.8*numel(gprice));

dataTrain = gprice(1:numTimeStepsTrain+1);
dataTest = gprice(numTimeStepsTrain+1:end);

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 150;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

%%
dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;

YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2))
mabe = mean(abs(YPred-YTest))
mape = mean(abs((YPred-YTest)/YTest))

%%
figure
plot(dataTrain)
hold on
%idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(2088:2608,YPred,'.-')
hold on
plot(2088:2608,price(2088:2608),'Color',[.7,.7,.7])
hold off
xlabel("time")
ylabel("price")
title("Forecast")
legend(["Observed" "Forecast"])

%%
numTimeStepsTrain = floor(1*numel(gprice));

dataTrain = gprice(1:numTimeStepsTrain);
%dataTest = gprice(numTimeStepsTrain+1:end);

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 150;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

%%
%dataTestStandardized = (dataTest - mu) / sig;
%XTest = dataTestStandardized(1:end-1);

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = 50;
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;

%YTest = dataTest(2:end);
%rmse = sqrt(mean((YPred-YTest).^2))
%mabe = mean(abs(YPred-YTest))
%mape = mean(abs((YPred-YTest)/YTest))

%%
figure
plot(dataTrain(2500:2609))
hold on
%idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(111:160,YPred,'.-')
%hold on
%plot(2610:2614,price(2610:2616),'Color',[.7,.7,.7])
hold off
xlabel("time")
ylabel("Gold Price (CAD/Oz)")
title("Actual Gold Price from 2019-10-30 + Forecast from 2020-04-01")
legend(["Observed" "Forecast"],'location','best')

%%
gp = bm(0,15.6551,'StartState',0);
[X, T] = gp.simByEuler(50,'nTrials', 500);
for i =1:500
    plot(T, X(:,:,i));
    hold on
end
xlabel("time");
ylabel("Gold Price (CAD/Oz)");
title("Simulated Differenced Gold Price");
hold off;

%%
exp = price(end);
plist = zeros(50,1);
dplist = zeros(50,1);
for i=2:51
    exdp = mean(X(i,:,:));
    exp = exp + exdp;
    plist(i-1) = exp;
    dplist(i-1) = exdp;
end

%%
figure
plot(price(2500:2609))
hold on
plot(111:160,plist,'.-')
hold off
xlabel("time")
ylabel("Gold Price (CAD/Oz)")
title("Actual Gold Price from 2019-10-30 + Forecast from 2020-04-01")
legend(["Observed" "Forecast"],'location','best')

%%
ff = [2224.883 2136.294 2143.012 2211.603 2228.557 2224.304...
    2224.846 2306.067 2357.866 2329.294 2258.644 2211.563...
    2245.067 2259.836 2204.013 2140.385 2180.329 2265.248 2319.364 ...
    2295.833 2275.253 2323.051 2345.158 2301.708 2194.655 ...
    2166.868 2217.029 2264.406 2223.675 2184.943 2246.748 ...
    2332.370 2357.412 2284.817 2241.572 2273.499 2307.911 ...
    2265.431 2172.683 2163.592 2234.987 2308.210 2279.539 2242.167...
    2291.773 2351.261 2349.017 2248.873 2189.493 2228.629];

%%
figure
plot(price(2500:2609))
hold on
plot(111:160,ff,'.-')
hold off
xlabel("time")
ylabel("Gold Price (CAD/Oz)")
title("Actual Gold Price from 2019-10-30 + Forecast from 2020-04-01")
legend(["Observed" "Forecast"],'location','best')