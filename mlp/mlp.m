function mlp()
    fprintf(1, "------------------------------------- \n");
    fprintf(1, "\x7c                MLP                \x7c\n");
    fprintf(1, "------------------------------------- \n");
    [P, T] = leerValoresDeArchivos();
    [P_train, T_train, P_val, T_val, P_test, T_test] = separarDatos(P, T);
    %[a, b] = leerRango();
    a = -2;
    b = 2;
    %[v1, v2] = leerCapas();
    v1 = [1 7 5 1];
    v2 = [2 3 1];
    %alpha = leerAlpha();
    alpha = 0.15;
    %[epochmax, eepoch, epochval, numval] = leerValoresParaCriterios();
    epochmax = 100;
    eepoch = 0.0001;
    epochval = 10;
    numval = 5;
    iniciarAprendizaje(P_train, T_train, P_val, T_val, P_test, T_test, alpha, v1, v2, epochmax, eepoch, epochval, numval);
end

function [P, T] = leerValoresDeArchivos()
    P = dlmread("p.txt");
    T = dlmread("targets.txt");
end

function [P_train, T_train, P_val, T_val, P_test, T_test] = separarDatos(P, T)
    fprintf(1, "Elija como deben estar separados los datos\n");
    fprintf(1, "1.- 70%% train - 15%% val - 15%% test\n");
    fprintf(1, "2.- 80%% train - 10%% val - 10%% test\n");
    opcion = input("Opcion: ");
    datos = size(P);
    fprintf(1, "cantidad de datos = %d\n", datos(1));
    P_train = [];
    T_train = [];
    P_val = [];
    T_val= [];
    P_test = [];
    T_test = [];
    if opcion == 1
        cant_train = round( datos(1)*0.7 );
        cant_val_test = round( datos(1)*0.15 );
        fprintf(1, "datos train = %d\n", cant_train);
        fprintf(1, "datos val = %d\n", cant_val_test);
        indices_train = randperm( datos(1), cant_train );
        indices_val = randperm( datos(1), cant_val_test );
        indices_test = randperm( datos(1), cant_val_test );
        % escogemos los datos de entrenamiento
        for i = 1:cant_train
            P_train(i,1) = P( indices_train(i) );
            T_train(i,1) = T( indices_train(i) );
        end
        % escogemos los datos de val y test
        for i = 1:cant_val_test
            P_val(i,1) = P( indices_val(i) );
            T_val(i,1) = T( indices_val(i) );
            P_test(i,1) = P( indices_test(i) );
            T_test(i,1) = T( indices_test(i) );
        end
    else
        cant_train = round( datos(1)*0.8 );
        cant_val_test = round( datos(1)*0.1 );
        fprintf(1, "datos train = %d\n", cant_train);
        fprintf(1, "datos val = %d\n", cant_val_test);
        indices_train = randperm( datos(1), cant_train );
        indices_val = randperm( datos(1), cant_val_test );
        indices_test = randperm( datos(1), cant_val_test );
        % escogemos los datos de entrenamiento
        for i = 1:cant_train
            P_train = P( indices_train(i) );
            T_train = T( indices_train(i) );
        end
        % escogemos los datos de val y test
        for i = 1:cant_val_test
            P_val(i,1) = P( indices_val(i) );
            T_val(i,1) = T( indices_val(i) );
            P_test(i,1) = P( indices_test(i) );
            T_test(i,1) = T( indices_test(i) );
        end
    end
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

function EEPOCH = calcularErrorEpoca(nro_ps, errores_iteracion)
    EEPOCH = zeros(nro_ps, 1);
    for i = 1:nro_ps
        EEPOCH = EEPOCH + errores_iteracion{i};
    end
    EEPOCH = abs(EEPOCH)/nro_ps;
end

function cumple = comprobarEEPOCH(EEPOCH, eepoch)
    tamanio = size(EEPOCH);
    cumple = 0;
    for i = 1:tamanio(1)
        cumple = cumple + ( EEPOCH(i,1) < eepoch );
    end
    if cumple == tamanio
        cumple = 1;
    end
end

function criterioAlcanzado = comprobarCF(epoch_actual, epochmax, EEPOCH, eepoch, nro_val, epochval)
    criterioAlcanzado = 0;
    if epoch_actual == epochmax
        fprintf(1, "Fin del entrenamiento. EPOCHMAX alcanzado\n");
        criterioAlcanzado = 1;
    else
        cumple = comprobarEEPOCH(EEPOCH, eepoch);
        if cumple == 1
            fprintf(1, "Fin del entrenamiento. Se cumplio eepoch\n");
            criterioAlcanzado = 1;
        else
            if nro_val == epochval
                fprintf(1, "Fin del entrenamiento. epochval alcanzado\n");
                criterioAlcanzado = 1;
            end
        end
    end
end

function iniciarAprendizaje(P_train, T_train, P_val, T_val, P_test, T_test, alpha, v1, v2, epochmax, eepoch, epochval, numval)
    % VALORES ALEATORIOS PARA CADA W y b
    % 
    % formula para nros aleatorios en un rango [a, b]
    % r = a + (b-a)*rand(N,1)
    nro_ps_train = size(P_train);
    nro_ps_val = size(P_val);
    
    val_errores = {};
    % cont_val incrementa si se verifica que val(k+1) > val(k)
    cont_val = 0;
    % nro_val es el contador de epocas de validaciones realizadas
    nro_val = 1;
    
    nro_capas = size(v1);
    capas_w_b = {};
    for c = 1:nro_capas(2)-1
        objeto = LayerMLP;
        objeto.w =  -1 + 2*rand( v1(c+1), v1(c) );
        objeto.b = -1 + 2*rand( v1(c+1), 1 );
        capas_w_b{c} = objeto;
    end
    
    % INICIO DE ENTRENAMIENTO
    for epoch_actual = 1:epochmax
        EEPOCH = [];
        if epoch_actual~=epochval && mod(epoch_actual, epochval)~=0
            fprintf(1, "ENTRENEMIENTO = EPOCA %d\n", epoch_actual);
            % EPOCA DE ENTRENAMIENTO
            arreglo_a = {};
            errores_iteracion = {};
            % VALIDACION DE P
            for nro_p = 1:nro_ps_train(1)
                p = P_train(nro_p, :)';
                target = T_train(nro_p, :)';
                % CALCULO DE 'a' CAPA POR CAPA
                arreglo_a{1} = p;
                for capa = 1:nro_capas(2)-1
                    n = (capas_w_b{capa}.w)*p + capas_w_b{capa}.b;
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
                F = calcularFs(v1, v2, arreglo_a);
                S = calcularSensitividades(nro_capas, F, e, capas_w_b);
                for j = 1:nro_capas(2)-1
                    w_old = capas_w_b{j}.w;
                    b_old = capas_w_b{j}.b;
                    capas_w_b{j}.w = calcularNuevoW(j, w_old, alpha, S, arreglo_a);
                    capas_w_b{j}.b = calcularNuevoBias(j, b_old, alpha, S);
                end
                % fin del CALCULO DE NUEVOS W y b
                
            end
            % fin de VALIDACION DE TODOS LOS P's
            EEPOCH = calcularErrorEpoca(nro_ps_train(1), errores_iteracion);
            %cumple = comprobarEEPOCH(EEPOCH, eepoch);
            % fin de EPOCA DE ENTRENAMIENTO
            fprintf(1, "ERROR DE ENTRENAMIENTO = %d\n", EEPOCH(1));
        else
            % EPOCA DE VALIDACION
            fprintf(1, "VALIDACION = EPOCA %d\n", epoch_actual);
            val_errores_iteracion = {};
            % VALIDACION DE P
            for nro_p_val = 1:nro_ps_val(1)
                p = P_val(nro_p_val, :)';
                target = T_val(nro_p_val, :)';
                
                for capa = 1:nro_capas(2)-1
                    n = (capas_w_b{capa}.w)*p + capas_w_b{capa}.b;
                    a = calcularA(n, v2(capa));
                    p = [];
                    p = a;
                end
                % fin del calculo de 'a'
                error_val = calcularError(target, a);
                
                % GUARDAMOS EL ERROR DE LA ITERACION (ENTRENAMIENTO)
                val_errores_iteracion{nro_p_val} = error_val;
            end
            % fin de VALIDACION DE TODOS LOS P's
            EEPOCH = calcularErrorEpoca(nro_ps_val(1), val_errores_iteracion);
            val_errores{nro_val} = EEPOCH;
            
            if nro_val >= 2
                aux = val_errores{nro_val} > val_errores{nro_val-1};
                suma = sum(aux);
                % si la suma es igual a nro_ps_val, entonces se cumple que
                % val_errores{nro_val} > val_errores{nro_val-1}
                if suma == nro_ps_val
                    cont_val = cont_val + 1;
                    if cont_val == numval
                        fprintf(1, "Fin del entrenamiento. numval alcanzado\n");
                        break;
                    end
                else
                    cont_val = 0;
                end
            end
            nro_val = nro_val + 1;
            % fin de EPOCA DE VALIDACION
            fprintf(1, "ERROR DE VALIDACION = %d\n", EEPOCH(1));
        end
        
        % VERIFICAMOS SI SE CUMPLE ALGUN CRITERIO DE FINALIZACION
        flag = comprobarCF(epoch_actual, epochmax, EEPOCH, eepoch, nro_val, epochval);
        if flag == 1
            break;
        end
        % fin de la verificacion de criterio de finalizacion
    end
    % fin de ENTRENAMIENTO
    
    % INICIO DEL TEST
    nro_ps_test = size(P_test);
    test_errores_iteracion = {};
    
    for i_test = 1:nro_ps_test(1)
        p = P_val(i_test, :)';
        target = T_val(i_test, :)';
                
        for capa = 1:nro_capas(2)-1
            n = (capas_w_b{capa}.w)*p + capas_w_b{capa}.b;
            a = calcularA(n, v2(capa));
            p = [];
            p = a;
        end
        % fin del calculo de 'a'
        e_p_test = calcularError(target, a);
        
        % GUARDAMOS EL ERROR DE LA ITERACION (ENTRENAMIENTO)
        test_errores_iteracion{i_test} = e_p_test;
    end
    
    % error_test debe estar en el rango [1x10^-3, 1x10^-4]
    error_test = calcularErrorEpoca(nro_ps_test(1), test_errores_iteracion);
    fprintf(1, "ERROR_TEST = %d \n", error_test(1));
    % fin del TEST
    objeto = LayerMLP;
    for c = 1:nro_capas(2)-1
        objeto = capas_w_b{c};
        disp(objeto.w);
        disp(objeto.b);
    end
end














