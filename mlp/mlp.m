function mlp()
    fprintf(1, "------------------------------------- \n");
    fprintf(1, "\x7c                MLP                \x7c\n");
    fprintf(1, "------------------------------------- \n");
    
end

function [P, T] = readValuesFromFiles()
    P = dlmread("p.txt");
    T = dlmread("targets.txt");
end

function [a, b] = readRange()
    fprintf(1, "Rango de la señal\n");
    a = input("Ingrese el limite inferior: ");
    b = input("Ingrese el limite superior: ");
end

function [v1, v2] = readLayers()
    capas = input("Ingrese el numero de capas de la red: ");
    v1 = [1];
    for i = 1:capas
        fprintf(1, "CAPA #%d\n", i);
        v1(i+1) = input("Numero de neuronas (25 max): ");
        v2(i) = input("Funcion de transferencia (1-3): ");
    end
end

function alfa = readAlpha()
    % Recomendacion: usar alfa = [0.1, 1x10^-4]
    alfa = input("Ingrese el valor de alfa: ");
end

















