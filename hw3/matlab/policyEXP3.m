classdef policyEXP3 < Policy
    %POLICYEXP3 This is a concrete class implementing EXP3.
    
    properties
        % Define member variables
    end
    
    methods

        function init(self, nbActions)
            % Initialize member variables
        end
        
        function action = decision(self)
            % Choose an action
        end
        
        function getReward(self, reward)
            % reward is the reward of the chosen action
            % update internal model
        end        
    end
end