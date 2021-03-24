classdef gameLookupTable < Game
    %GAMELOOKUPTABLE This is a concrete class defining a game defined by an
    %external input
    
    methods
        function self = gameLookupTable(tabInput, isLoss)
            % Input
            %   tabInput - input table (actions x rewards or losses)
            %   isLoss - 1 if input table represent loss, 0 otherwise
        end
        
    end
    
end
