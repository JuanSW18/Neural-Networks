function mlp()
    fprintf(1, "------------------------------------- \n");
    fprintf(1, "\x7c                MLP                \x7c\n");
    fprintf(1, "------------------------------------- \n");
    leerCapas();
end

function [P, T] = leerValoresDeArchivos()
    P = dlmread("p.txt");
    T = dlmread("targets.txt");
end

function [a, b] = leerRango()
    fprintf(1, "Rango de la señal\n");
    a = input("Ingrese el limite inferior: ");
    b = input("Ingrese el limite superior: ");
end

function [v1, v2] = leerCapas()
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

function alfa = leerAlpha()
    % Recomendacion: usar alfa = [0.1, 1x10^-4]
    alfa = input("Ingrese el valor de alfa: ");
end

function [epochmax, eepoch, epochval, numval] = leerValoresParaCriterios()
    % epochval => 10% de epochmax
    epochmax = input("Ingrese el numero maximo de epocas: ");
    eepoch = input("Ingrese eepoch: ");
    epochval = input("Ingrese epochval: ");
    numval = input("Ingrese numval: ");
end

function a = calcularA(n, ft)
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

function e = calcularError(t, a)
    [filas, columnas] = size(t);
    e = t - a;
end

function F = calcularFs(v1, v2, valores_a)
    % valores_a = {a0=p, a1, a2, a3}
    F = {};    
    nro_capas = size(v1);
    for i = 2:nro_capas
        f = zeros(v1(i), v1(i));
        switch v2(i-1)
            case 1
                % purelin
                for j = 1:v1(i)
                    f(j, j) = 1;
                end
            case 2
                % logsig
                 for j = 1:v1(i)
                    f(j, j) = valores_a{i}(j, 1)*(1 - valores_a{i}(j, 1));
                end
            case 3
                % tansig
                 for j = 1:v1(i)
                    f(j, j) = 1 - valores_a{i}(j, 1)*valores_a{i}(j, 1);
                end
        end   
    end 
end

function S = calcularSensitividades(nro_capas, F, e, capas_w_b)
    S = {};
    for n = 1:nro_capas
        S{n} = 0;
    end
    S{nro_capas} = -2*F(nro_capas)*e;
    for i = nro_capas-1:-1:1
        S{i} = F(i)*(capas_w_b{i+1}.w)'*S{i+1};
    end
end

function w = calcularNuevoW(nro_capas, w_old, alpha, S, arreglo_a)
    for i = 2:nro_capas
        w = w_old - alpha*S{i}*arreglo_a{i-1};
    end
end

function b = calcularNuevoBias(nro_capas, b_old, alpha, S)
    for i = 1:nro_capas
        b = b_old - alpha*S{i};
    end
end

function iniciarAprendizaje(P, T, alpha, v1, v2, epochmax, epochval)
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
            arreglo_a = {};
            errores_iteracion = {};
            % VALIDACION DE P
            for nro_p = 1:nro_ps
                p = P(nro_p, :)';
                target = T(nro_p, :)';
                % CALCULO DE 'a' CAPA POR CAPA
                arreglo_a{1} = p;
                for capa = 1:nro_capas(2)-1
                    n = capas_w_b{capa}.w * p + capas_w_b{capa}.b;
                    a = calcularA(n, v2(capa));
                    p = [];
                    p = a;
                    
                    % GUARDAMOS EL VALOR FINAL DE a PARA CADA CAPA
                    arreglo_a{capa+1} = a;
                end
                % fin del calculo de 'a'
                e = calcularError(target, a);
                
                % GUARDAMOS EL ERROR DE LA ITERACION (ENTRENAMIENTO)
                errores_iteracion{nro_p} = e;
                
                % CALCULO DE NUEVOS W y b
                F = calcularFs(v1, v2, a);
                S = calcularSensitividades(nro_capas, F, e, capas_w_b);
                for j = 1:nro_capas(2)-1
                    w_old = capas_w_b{j}.w;
                    b_old = capas_w_b{j}.b;
                    capas_w_b{j}.w = calcularNuevoW(nro_capas, w_old, alpha, S, arreglo_a);
                    capas_w_b{j}.b = calcularNuevoBias(nro_capas, b_old, alpha, S);
                end
                % fin del CALCULO DE NUEVOS W y b
                
            end
            % fin de VALIDACION DE TODOS LOS P's
            
            % fin de EPOCA DE ENTRENAMIENTO
        else
            % EPOCA DE VALIDACION
            
            
            % fin de EPOCA DE VALIDACION
        end
    end
end

% CALCULO DE NUEVOS W y b
%                 for j = 1:nro_capas(2)-1
%                     w_old = capas_w_b{j}.w;
%                     b_old = capas_w_b{j}.b;
%                     capas_w_b{j}.w = calculateNewW(w_old, p, e, alpha);
%                     capas_w_b{j}.b = calculateNewBias(w_old, e, alpha);
%                 end















