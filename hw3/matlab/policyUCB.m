classdef policyUCB < Policy
    %POLICYUCB This is a concrete class implementing UCB.

        
    properties
        % Member variables
    end
    
    methods
        function init(self, nbActions)
            % Initialize
        end
        
        function action = decision(self)
            % Choose action
        end
        
        function getReward(self, reward)
            % Update ucb
        end        
    end

end
