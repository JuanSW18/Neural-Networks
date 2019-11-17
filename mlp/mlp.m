function mlp()
    fprintf(1, "------------------------------------- \n");
    fprintf(1, "\x7c                MLP                \x7c\n");
    fprintf(1, "------------------------------------- \n");
    readLayers();
end

function [P, T] = readValuesFromFiles()
    P = dlmread("p.txt");
    T = dlmread("targets.txt");
end

function [a, b] = readRange()
    fprintf(1, "Rango de la se�al\n");
    a = input("Ingrese el limite inferior: ");
    b = input("Ingrese el limite superior: ");
end

function [v1, v2] = readLayers()
    capas = input("Ingrese el numero de capas de la red (max. 3): ");
    v1 = [1];
    for i = 1:capas
        fprintf(1, "CAPA #%d\n", i);
        v1(i+1) = input("Numero de neuronas (25 max): ");
        v2(i) = input("Funcion de transferencia (1-3): ");
    end
    v1(i+2) = 1;
    v2(i+1) = 1;
end

function alfa = readAlpha()
    % Recomendacion: usar alfa = [0.1, 1x10^-4]
    alfa = input("Ingrese el valor de alfa: ");
end

function [epochmax, eepoch, epochval, numval] = criterias()
    % epochval => 10% de epochmax
    epochmax = input("Ingrese el numero maximo de epocas: ");
    eepoch = input("Ingrese eepoch: ");
    epochval = input("Ingrese epochval: ");
    numval = input("Ingrese numval: ");
end

function startLearning(v1, v2, epochmax, epochval)
    % VALORES ALEATORIOS PARA CADA W y b
    % 
    % formula para nros aleatorios en un rango [a, b]
    % r = a + (b-a)*rand(N,1)
    tam = size(v1);
    capas_w_b = [LayerMLP, LayerMLP, LayerMLP];
    for i = 1:tam+2
        objeto = LayerMLP;
        objeto.w =  -1 + 2*rand( v1(i+1), v1(i) );
        objeto.b = -1 + 2*rand( v1(i+1), 1 );
        capas_w_b(i) = objeto;
    end
    
    for epoch_actual = 1:epochmax
        if epoch_actual~=epochval || mod(epoch_actual, epochval)~=0
            % EPOCA DE ENTRENAMIENTO
        else
            % EPOCA DE VALIDACION
        end
    end
end















