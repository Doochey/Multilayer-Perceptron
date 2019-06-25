close;
clear;
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
input = images(:,1:10000);


online = MLP2(size(input, 1), 100, 64, 10);
online = online.initWeight(1.0);


epoch = 0;
Ovalcorrect = 0;
Otcorrect = 0;
Bvalcorrect = 0;
Btcorrect = 0;

% Test Online Algorithm
while (online.oldPercent < 100 && epoch < 100)
    
    % Train MLP
    epoch = epoch + 1
    online = online.online_adapt(input, labels, 0.001);
    percentCorrect = online.oldPercent;
    if epoch > 1
        line([epoch-1 epoch],[prePercentCorrect, percentCorrect], 'Color', 'red', 'LineStyle','-');
        drawnow;
        hold on;
    end
    prePercentCorrect = percentCorrect;
    
    % Get Validation set
    val = images(:,50001:60000);

    % Compute Validation error rate
    valOutput = online.compute_output(val);
    numberCorrect = 0;
    for col=1:size(val,2)
        expectedValue = labels(col+50000);
        class = 11;
        max = 0;
        for row=1:size(valOutput,1)
            if (valOutput(row, col) > max)
                class = row;
                max = valOutput(row, col);
            end
        end
        if (class == 10)
            class = 0;
        end
        if (class == expectedValue)
            numberCorrect = numberCorrect + 1;
        end
    end
    valCorrect = numberCorrect / size(val,2) * 100;
    verrorRate = 100 - valCorrect
    Ovalcorrect = valCorrect;
    
    % Load test set
    timages = loadMNISTImages('t10k-images.idx3-ubyte');
    tlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
    tinput = timages(:,1:10000);

    % Compute test error rate
    toutput = online.compute_output(tinput);
    numberCorrect = 0;
    for col=1:size(tinput,2)
        expectedValue = tlabels(col);
        class = 11;
        max = 0;
        for row=1:size(toutput,1)
            if (toutput(row, col) > max)
                class = row;
                max = toutput(row, col);
            end
        end
        if (class == 10)
            class = 0;
        end
        if (class == expectedValue)
            numberCorrect = numberCorrect + 1;
        end
    end
    tCorrect = numberCorrect / size(tinput,2) * 100;
    terrorRate = 100 - tCorrect
    Otcorrect = tCorrect;
end

images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
input = images(:,1:10000);
batch = MLP2(size(input, 1), 100, 64, 10);
batch = batch.initWeight(1.0);

epoch = 0;

% Test Batch Algorithm
while (batch.percentageCorrect < 100 && epoch < 100)
    
    % Train MLP
    epoch = epoch + 1
    batch = batch.batch_adapt(input, labels, 0.001);
    percentCorrect = batch.percentageCorrect;
    if epoch > 1
        line([epoch-1 epoch],[prePercentCorrect, percentCorrect], 'Color', 'blue', 'LineStyle','-');
        drawnow;
        hold on;
    end
    prePercentCorrect = percentCorrect;
    
    % Get Validation set
    val = images(:,50001:60000);

    % Compute Validation error rate
    valOutput = batch.compute_output(val);
    numberCorrect = 0;
    for col=1:size(val,2)
        expectedValue = labels(col+50000);
        class = 11;
        max = 0;
        for row=1:size(valOutput,1)
            if (valOutput(row, col) > max)
                class = row;
                max = valOutput(row, col);
            end
        end
        if (class == 10)
            class = 0;
        end
        if (class == expectedValue)
            numberCorrect = numberCorrect + 1;
        end
    end
    valCorrect = numberCorrect / size(val,2) * 100;
    verrorRate = 100 - valCorrect
    Bvalcorrect = valCorrect;
    
    % Load test set
    timages = loadMNISTImages('t10k-images.idx3-ubyte');
    tlabels = loadMNISTLabels('t10k-labels.idx1-ubyte');
    tinput = timages(:,1:10000);

    % Compute test error rate
    toutput = batch.compute_output(tinput);
    numberCorrect = 0;
    for col=1:size(tinput,2)
        expectedValue = tlabels(col);
        class = 11;
        max = 0;
        for row=1:size(toutput,1)
            if (toutput(row, col) > max)
                class = row;
                max = toutput(row, col);
            end
        end
        if (class == 10)
            class = 0;
        end
        if (class == expectedValue)
            numberCorrect = numberCorrect + 1;
        end
    end
    tCorrect = numberCorrect / size(tinput,2) * 100;
    terrorRate = 100 - tCorrect
    Btcorrect = tCorrect;
end

online.oldPercent
Ovalcorrect
Otcorrect

batch.percentageCorrect
Bvalcorrect
Btcorrect


hold off;

