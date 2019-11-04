%
% ADALINE
% 
% Modo clasificador y modo regresor
% Se debera ingresar 2 archivos: p.txt y targets.txt
% epochmax: Se refiere al número máximo de épocas a realizar
% Eepoch: es un valor pequeño al cuál se desea que llegue el error 
% promedio de todos los datos del conjunto de entrenamiento
%
function adaline()
    fprintf(1, "------------------------------------- \n");
    fprintf(1, "\x7c              ADALINE              \x7c\n");
    fprintf(1, "------------------------------------- \n");
    [P, T] = readValuesFromFiles();
    
    epochmax = input("Ingrese el valor de epochmax: ");
    Eepoch = input("Ingrese el valor de Eepoch: ");
    alfa = input("Ingrese el valor del factor de aprendizaje: ");
    modo = input("Ingrese R(Regresor) o C(Clasificador) segun el tipo de problema: ", 's');
    if modo == "r" || modo == "R"
        fprintf(1, "Empezando regresion ...\n");
        initRegression(P, T, epochmax, Eepoch, alfa);
    else
        if modo == "c" || modo == "C"
            fprintf(1, "Empezando clasificacion ...\n");
            initClasification(P, T, epochmax, Eepoch, alfa);
        else
            fprintf(1, "Letra incorrecta\n");
            adaline();
        end
    end
end

% Leer archivos para el dataset
function [P, T] = readValuesFromFiles()
    P = dlmread("p_reg.txt");
    T = dlmread("targets_reg.txt");
end

% Calcular el numero de clases
function S = calculateS(T)
    [filas, columnas] = size(T);
    classes_number = 0;
    % asumiendo que los targets iguales estan uno tras otro
    for i = 2:filas
        f1_aux = T(i, :);
        f2_aux = T(i-1, :);
        if ~isequal(f1_aux, f2_aux)
            classes_number = classes_number + 1;
        end
    end
    % 2^s = nro_clases => s = log2(nro_clases)
    S = ceil(log2(classes_number));
end

% Imprimir el modelo matematico
% a = purelin(W*p + b)
% S*1      S*R R*1 S*1 
% function printMathematicModel()
%     fprintf(1, "...\n");
% end

function eepoch = getEEPOCH(e)
    [filas, columnas] = size(e);
    eepoch = zeros(filas, 1);
    for i = 1: columnas
        eepoch = eepoch + e(:, i) / columnas;
    end
    eepoch = abs(eepoch);
end

function [W, b] = getNewValues(W_ant, b_ant, p, e, alfa)
    W = W_ant + ( 2 * alfa )* e * p';
    b = b_ant + ( 2 * alfa )* e ;
end

function initClasification(P, T, epochmax, Eepoch, alfa)
    % VALORES ALEATORIOS PARA W y b
    % W tiene una dimension de SxR
    % obtendremos R del vector P
    % 
    % formula para nros aleatorios en un rango [a, b]
    % r = a + (b-a)*rand(N,1)
    [filas, columnas] = size(P);
    S = calculateS(T);
    W = -1 + ( 1 + 1 )*rand( S, columnas );
    b = -1 + ( 1 + 1 )*rand( S, 1 );
    
    %W = [1 0; 0 1];
    %b = [1; 1];
    
    % Arreglo de EEPOCH's
    a_EEPOCH = [];
    
    % APRENDIZAJE
    for i = 1:epochmax
        fprintf(1, "EPOCA %d\n", i);
        % Arreglo de errores
        a_error = zeros( columnas, filas );
        for j = 1:filas
            disp(W);
            disp(b);
            % cada p esta escrito de forma traspuesta en el archivo
            % por lo que debemos aplicar la traspuesta
            p = P(j, :)';
            n = W * p + b;
            a = purelin(n);
            
            % CALCULO DEL ERROR
            a_error(:, j) = ( T(j, :)' - a );
            fprintf(1, "Error:\n");
            disp(a_error(:, j));
            % CALCULO DE NUEVOS VALORES PARA W y b
            [W, b] = getNewValues(W, b, p, a_error(:, j), alfa);
            fprintf(1, "-----------------------------------\n");
        end
        EEPOCH = getEEPOCH(a_error);
        
        fprintf(1, "EEPOCH:\n");
        disp(EEPOCH);
        fprintf(1, "\n\n");
        
        a_EEPOCH(:, i) = EEPOCH;
        % VERIFICANDO SI EEPOCH ES CERO
        suma_EEPOCH = 0;
        for n = 1: columnas
            suma_EEPOCH = suma_EEPOCH + ( EEPOCH(n, 1) == 0);
        end
        if suma_EEPOCH == columnas
            fprintf(1, "Aprendizaje finalizado. EEPOCH = 0\n");
            break;
        else
            menores = 0;
            for n = 1: columnas
                menores = menores + ( EEPOCH(n, 1) < Eepoch);
            end
            if menores == columnas
                fprintf(1, "Aprendizaje finalizado. EEPOCH < Eepoch\n");
                break;
            else
                if i == epochmax
                    fprintf(1, "Aprendizaje finalizado. epochmax alcanzado\n");
                    break;
                end
            end
        end
    end
end

function W = getNewW(W_ant, p, e, alfa)
    W = W_ant + ( 2 * alfa )* e * p';
end

function initRegression(P, T, epochmax, Eepoch, alfa)
    [filas, columnas] = size(P);
    W = -1 + ( 1 + 1 )*rand( 1, columnas );
    W = [0.84 0.39 0.78];
    % Arreglo de EEPOCH's
    a_EEPOCH = [];
    
    % APRENDIZAJE
    for i = 1:epochmax
        fprintf(1, "EPOCA %d\n", i);
        % Arreglo de errores
        a_error = zeros( 1, filas );
        for j = 1:filas
            fprintf(1, "\t d%d\n", j);
            disp(W);
            
            p = P(j,:)';
            n = W * p;
            a = purelin(n);
            
            % CALCULO DEL ERROR
            a_error(1,j) = ( T(j, 1) - a );
            fprintf(1, "\tERROR=%.4f\n", a_error(1,j));
            
            % CALCULO DEL NUEVO VALOR PARA W
            W = getNewW(W, p, a_error(1, j), alfa);
        end
        EEPOCH = getEEPOCH(a_error);
        
        fprintf(1, "\tEEPOCH=%.4f\n\n", EEPOCH);
        
        a_EEPOCH(:, i) = EEPOCH;
        
        % VERIFICANDO SI EEPOCH ES CERO
        if EEPOCH == 0
            fprintf(1, "Aprendizaje finalizado. EEPOCH = 0\n");
            break;
        else
            if EEPOCH < Eepoch
                fprintf(1, "Aprendizaje finalizado. EEPOCH < Eepoch\n");
                break;
            else
                if i == epochmax
                    fprintf(1, "Aprendizaje finalizado. epochmax alcanzado\n");
                    break;
                end
            end
        end
    end
end





