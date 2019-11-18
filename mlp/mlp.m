function mlp()
    fprintf(1, "------------------------------------- \n");
    fprintf(1, "\x7c                MLP                \x7c\n");
    fprintf(1, "------------------------------------- \n");
    [P, T] = leerValoresDeArchivos();
    %[a, b] = leerRango();
    a = -2;
    b = 2;
    %[v1, v2] = leerCapas();
    v1 = [1 2 3 1];
    v2 = [2 3 1];
    %alpha = leerAlpha();
    alpha = 0.15;
    %[epochmax, eepoch, epochval, numval] = leerValoresParaCriterios();
    epochmax = 15;
    eepoch = 0.001;
    epochval = 10;
    numval = 5;
    iniciarAprendizaje(P, T, alpha, v1, v2, epochmax, epochval);
end

function [P, T] = leerValoresDeArchivos()
    P = dlmread("p.txt");
    T = dlmread("targets.txt");
end

function [a, b] = leerRango()
    fprintf(1, "Rango de la se�al\n");
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
    for i = 2:nro_capas(2)
        f = zeros(v1(i), v1(i));
        switch v2(i-1)
            case 1
                % purelin
                for j = 1:v1(i)
                    f(j, j) = 1;
                end
            case 2
                % logsig
                aux =  cell2mat( valores_a(1, i) );
                for j = 1:v1(i)
                    f(j, j) = aux(j, 1)*(1 - aux(j, 1));
                end
            case 3
                % tansig
                aux =  cell2mat( valores_a(1, i) );
                for j = 1:v1(i)
                    f(j, j) = 1 - aux(j, 1)*aux(j, 1);
                end
        end
        F{i-1} = f;
    end 
end

function S = calcularSensitividades(nro_capas, F, e, capas_w_b)
    S = {};
    for n = 1:nro_capas(2)-1
        S{n} = 0;
    end
    S{nro_capas(2)-1} = -2*F{nro_capas(2)-1}*e;
    for i = nro_capas-2:-1:1
        S{i} = F{i}*(capas_w_b{i+1}.w)'*S{i+1};
    end
end

function w = calcularNuevoW(capa, w_old, alpha, S, arreglo_a)
    w = w_old - alpha*S{capa}*arreglo_a{capa}';
%     for i = 2:nro_capas
%         w = w_old - alpha*S{i}*arreglo_a{i-1}';
%     end
end

function b = calcularNuevoBias(capa, b_old, alpha, S)
    b = b_old - alpha*S{capa};
%     for i = 1:nro_capas
%         b = b_old - alpha*S{i};
%     end
end

function iniciarAprendizaje(P, T, alpha, v1, v2, epochmax, epochval)
    % VALORES ALEATORIOS PARA CADA W y b
    % 
    % formula para nros aleatorios en un rango [a, b]
    % r = a + (b-a)*rand(N,1)
    nro_ps = size(P);
    
    nro_capas = size(v1);
    capas_w_b = {};
    for c = 1:nro_capas(2)-1
        objeto = LayerMLP;
        objeto.w =  -1 + 2*rand( v1(c+1), v1(c) );
        objeto.b = -1 + 2*rand( v1(c+1), 1 );
        capas_w_b{c} = objeto;
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
                fprintf(1, "P %d\n", nro_p);
                for capa = 1:nro_capas(2)-1
%                     fprintf(1, "CAPA %d\n", capa);
%                     fprintf(1, "W\n");
%                     disp(capas_w_b{capa}.w);
%                     fprintf(1, "p\n");
%                     disp(p);
%                     fprintf(1, "bias\n");
%                     disp(capas_w_b{capa}.b);
                    n = (capas_w_b{capa}.w)*p + capas_w_b{capa}.b;
                    a = calcularA(n, v2(capa));
                    p = [];
                    p = a;
%                     fprintf(1, "a\n");
%                     disp(a);
                    % GUARDAMOS EL VALOR FINAL DE a PARA CADA CAPA
                    arreglo_a{capa+1} = a;
                end
                % fin del calculo de 'a'
                e = calcularError(target, a);
                
                % GUARDAMOS EL ERROR DE LA ITERACION (ENTRENAMIENTO)
                errores_iteracion{nro_p} = e;
                
                % CALCULO DE NUEVOS W y b
                F = calcularFs(v1, v2, arreglo_a);
                S = calcularSensitividades(nro_capas, F, e, capas_w_b);
%                 fprintf(1, "Nuevos w y b\n");
                for j = 1:nro_capas(2)-1
                    w_old = capas_w_b{j}.w;
                    b_old = capas_w_b{j}.b;
                    capas_w_b{j}.w = calcularNuevoW(j, w_old, alpha, S, arreglo_a);
                    capas_w_b{j}.b = calcularNuevoBias(j, b_old, alpha, S);
%                     fprintf(1, "j => %d\n", j);
%                     fprintf(1, "old values\n");
%                     disp(w_old);
%                     disp(b_old);
%                     fprintf(1, "new values\n");
%                     disp(capas_w_b{j}.w);
%                     disp(capas_w_b{j}.b);
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














