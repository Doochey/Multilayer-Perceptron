classdef MLP2
    % 3 layer perceptron
    properties (SetAccess=private)
        inputDim
        hiddenDim
        hidden2Dim
        outputDim
        
        hiddenWeights
        hidden2Weights
        outputWeights
        
        percentageCorrect
        mse
        numberCorrect
        oldPercent
    end
    
    methods
        function obj=MLP2(inputD,hiddenD,hidden2D,outputD)
            %obj=MLP
            obj.inputDim=inputD;    % Number of input dimensions
            obj.hiddenDim=hiddenD;  % Number of hidden neurons in layer 1
            obj.hidden2Dim = hidden2D; % Number of hidden neurons in layer 2
            obj.outputDim=outputD; % Number of output neurons
            obj.hiddenWeights=zeros(hiddenD,inputD+1);
            obj.outputWeights=zeros(outputD,hiddenD+1);
            obj.mse = 1;
            obj.percentageCorrect = 0;
            obj.oldPercent = 0;
            obj.numberCorrect = 0;
        end
        
        function obj=initWeight(obj,variance)
            v = variance;
            obj.hiddenWeights = v + (-v-v).*rand(obj.hiddenDim,obj.inputDim+1);
            obj.hidden2Weights = v + (-v-v).*rand(obj.hidden2Dim,obj.hiddenDim+1);
            obj.outputWeights = v + (-v-v).*rand(obj.outputDim,obj.hidden2Dim+1);
        end
	% forward propagation of values
        function [hiddenNet,hidden,hidden2Net, hidden2,outputNet,output]=compute_net_activation(obj, input)
            %Add bias Dimension to input
            bias = ones(1, size(input,2));
            input = [input;bias];
            hiddenNet = obj.hiddenWeights * input;  
            hidden = logsig(hiddenNet);
            
            bias = ones(1, size(hidden,2));
            hidden2input = [hidden;bias];
            hidden2Net = obj.hidden2Weights * hidden2input;  
            hidden2 = logsig(hidden2Net);
            
            
            bias = ones(1, size(hidden2,2));
            outputInput = [hidden2;bias];
            outputNet = obj.outputWeights * outputInput;
            output = logsig(outputNet);
        end
        
        function output=compute_output(obj,input)
            [hN,h,h2N,h2,oN,output] = obj.compute_net_activation(input);
        end
        
        
        
	% back propagation of errors
        function obj=adapt_to_target(obj,input,target,rate)
            
            
            expectedValues = zeros(obj.outputDim, size(input, 2));
            for row=1:size(target, 1)
                if (target(row) == 0)
                    expectedValues(10, row) = 1; % row 10 represents 0
                else 
                    expectedValues(target(row), row) = 1;
                end
            end
            
            [hN,h,h2N,h2,oN,output] = obj.compute_net_activation(input);
            errors = zeros(obj.outputDim, size(input, 2));
            squareErrors = zeros(obj.outputDim, size(input, 2));
            sumDeltaOutputWeights = zeros(obj.outputDim, obj.hidden2Dim+1);
            sumDeltaHiddenWeights = zeros(obj.hiddenDim, obj.inputDim+1);
            sumDeltaHidden2Weights = zeros(obj.hidden2Dim, obj.hiddenDim+1);
            for col=1:size(input,2)
               class = 11;
               max = 0;
               for row=1:size(output,1)
                   if (output(row, col) > max)
                       class = row;
                       max = output(row, col);
                   end
                   outputValue = output(row, col);
                   expectedValue = expectedValues(row, col);
                   difference =  expectedValue - outputValue;
                   errors(row, col) = difference;
                   squaredDifference = difference^2;
                   squareErrors(row, col) = squaredDifference;
               end
               if (class == 10)
                   class = 0;
               end
               if (class == target(col))
                   obj.numberCorrect = obj.numberCorrect + 1;
               end
               
               % Back propragate errors
               hidden2Errors(:, col) = transpose(obj.outputWeights) * errors(:, col);
               hiddenErrors(:, col) = transpose(obj.hidden2Weights) * hidden2Errors(1:size(hidden2Errors, 1) - 1, col);
               
               % Calculate desired weight changes
               % = Learning Rate * Errors * sigmoid derivative * input to
               % layer
               deltaOutputWeights = rate * errors(:, col) .* ((output(:, col) .* (1-output(:, col)))) * transpose(h2(:, col));
               deltaOutputBias = rate * errors(:, col) .* ((output(:, col) .* (1-output(:, col))));
               deltaOutputWeights = [deltaOutputWeights deltaOutputBias];
               sumDeltaOutputWeights = sumDeltaOutputWeights + deltaOutputWeights;
               
               deltaHidden2Weights = rate * hidden2Errors(1:obj.hidden2Dim, col) .* ((h2(:, col) .* (1-h2(:, col)))) * transpose(h(:,col));
               deltaHidden2Bias = rate * hidden2Errors(obj.hidden2Dim+1:obj.hidden2Dim+1, col) .* ((h2(:, col) .* (1-h2(:, col))));
               deltaHidden2Weights = [deltaHidden2Weights deltaHidden2Bias];
               sumDeltaHidden2Weights = sumDeltaHidden2Weights + deltaHidden2Weights;
               
               deltaHiddenWeights = rate * hiddenErrors(1:obj.hiddenDim, col) .* ((h(:, col) .* (1-h(:, col)))) * transpose(input(:,col));
               deltaHiddenBias = rate * hiddenErrors(obj.hiddenDim+1:obj.hiddenDim+1, col) .* ((h(:, col) .* (1-h(:, col))));
               deltaHiddenWeights = [deltaHiddenWeights deltaHiddenBias];
               sumDeltaHiddenWeights = sumDeltaHiddenWeights + deltaHiddenWeights;
               
               
            end
            % Nudge Weights
            obj.hiddenWeights = obj.hiddenWeights + sumDeltaHiddenWeights;
            obj.outputWeights = obj.outputWeights + sumDeltaOutputWeights;
            obj.hidden2Weights = obj.hidden2Weights + sumDeltaHidden2Weights;
            
            obj.mse = mean(squareErrors);
            obj.mse = mean(obj.mse);
            
        end
        
        function obj=online_adapt(obj,input,target,rate)
            for i=1:size(input, 2)
               obj = obj.adapt_to_target(input(:,i), target(i), rate); 
            end
            obj.percentageCorrect = (obj.numberCorrect / size(input, 2)) * 100
            obj.oldPercent = obj.percentageCorrect;
            obj.percentageCorrect = 0;
            obj.numberCorrect = 0;
        end
        
        
        
        function obj=batch_adapt(obj,input,target,rate)
            obj.percentageCorrect = 0;
            obj = obj.adapt_to_target(input, target, rate);
            obj.percentageCorrect = obj.numberCorrect / size(input, 2) * 100
            obj.numberCorrect = 0;
        end
        
    end
end


