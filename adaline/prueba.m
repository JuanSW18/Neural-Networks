function prueba()
    w1 = [1 -1 1 -1 1 -1 1 -1 -1 1 1; 1 -1 -1 1 -1 1 -1 1 -1 1 1];
    b1 = [-2 3 0.5 0.5 -1.75 2.25 -3.25 3.75 6.25 -5.75 -4.750];
    w2 = [1 1 1 1 0 0 0 0 0 0 0; 0 0 0 0 1 1 0 0 1 0 1; 0 0 0 0 1 0 0 1 1 1 0; 0 0 0 0 0 0 1 1 1 0 1];
    b2 = [-3; -3; -3; -3];
    w3 = [1 1 1 1];
    b3 = [3];
    
    p = [4.8; 1.35];
    disp(p);
    
    a1 = w1'*p + b1';
    disp(a1);
    a1 = hardlims(a1);

    a2 = w2*a1 + b2;
    disp(a2);
    a2 = hardlims(a2);

    a3 = w3*a2 + b3;
    disp(a3);
    a3 = hardlims(a3);
        
%     aux_x = p(1, 1);
%     aux_y = p(2, 1);
%     while p(1, 1) <= 4.35
%         p(2, 1) = aux_y;
%         while p(2, 1)  <= 1.95
%             a1 = w1'*p + b1';
%             %disp(a1);
%             a1 = hardlims(a1);
% 
%             a2 = w2*a1 + b2;
%             %disp(a2);
%             a2 = hardlims(a2);
% 
%             a3 = w3*a2 + b3;
%             %disp(a3);
%             a3 = hardlims(a3);
%             
%             if a3 < 0
%                 disp(p);
%             end
%             %disp(p);
%             p(2, 1) = p(2, 1) + 0.01;
%         end
%         p(1, 1) = p(1, 1) + 0.01;
%     end
end

