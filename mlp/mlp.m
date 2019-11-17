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
    fprintf(1, "Rango de la señal\n");
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

function a = calculateA(n, ft)
    [filas, columnas] = size(n);
    a = zeros(filas, columnas);
    switch ft
        case 1
            % purelin
            for i = 1:filas
                for j = 1:columnas
                    a(i, j) = purelin( n(i, j) );
                end
            end
        case 2
            % logsig
            for i = 1:filas
                for j = 1:columnas
                    a(i, j) = logsig( n(i, j) );
                end
            end
        case 3
            % tansig
            for i = 1:filas
                for j = 1:columnas
                    a(i, j) = tansig( n(i, j) );
                end
            end
    end
end

function e = calculateError(t, a)
    [filas, columnas] = size(t);
    e = zeros(filas, columnas);
    e = t - a;
end

function w = calculateW(w_old, p, e, alpha)
    w = w_old + 2*alpha*e*p';
end

function b = calculateBias(b_old, e, alpha)
    b = b_old + 2*alpha*e;
end

function startLearning(P, T, alpha, v1, v2, epochmax, epochval)
    % VALORES ALEATORIOS PARA CADA W y b
    % 
    % formula para nros aleatorios en un rango [a, b]
    % r = a + (b-a)*rand(N,1)
    nro_ps = size(P);
    
    nro_capas = size(v1);
    capas_w_b = {};
    for j = 1:nro_capas(2)-1
        objeto = LayerMLP;
        objeto.w =  -1 + 2*rand( v1(j+1), v1(j) );
        objeto.b = -1 + 2*rand( v1(j+1), 1 );
        capas_w_b{j} = objeto;
    end
    
    for epoch_actual = 1:epochmax
        if epoch_actual~=epochval || mod(epoch_actual, epochval)~=0
            % EPOCA DE ENTRENAMIENTO
            errores_iteracion = {};
            for nro_p = 1:nro_ps
                p = P(nro_p, :)';
                target = T(nro_p, :)';
                for capa = 1:nro_capas(2)-1
                    n = capas_w_b{capa}.w * p + capas_w_b{capa}.b;
                    a = calculateA(n, v2(capa));
                    p = [];
                    p = a;
                end
                e = calculateError(target, a);
                errores_iteracion{nro_p} = e;
                % CALCULO DE NUEVOS W y b
                for j = 1:nro_capas(2)-1
                    w_old = capas_w_b{j}.w;
                    b_old = capas_w_b{j}.b;
                    capas_w_b{j}.w = calculateW(w_old, p, e, alpha);
                    capas_w_b{j}.b = calculateBias(w_old, e, alpha);
                end
            end
        else
            % EPOCA DE VALIDACION
        end
    end
end















