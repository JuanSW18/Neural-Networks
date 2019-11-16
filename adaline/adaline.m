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
    P = dlmread("p_xor.txt");
    T = dlmread("targets_xor.txt");
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
    if classes_number == 1
        classes_number = 2;
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

function v = transformW(W)
    [filas, columnas] = size(W);
    k = 1;
    v = zeros(filas*columnas, 1);
    for i = 1:filas
        for j = 1:columnas
            v(k, 1) = W(i, j);
            k = k + 1;
        end
    end
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
    
    % Arreglo de EEPOCH's, W's y b's
    a_EEPOCH = [];
    a_W = [];
    a_b = [];
    t = 1;
    % APRENDIZAJE
    for i = 1:epochmax
        % Arreglo de errores
        a_error = zeros( S, filas );
        for j = 1:filas
            % GUARDAMOS VALORES DE W y b
            aux = transformW(W);
            
            a_W(:, t) = aux;
            a_b(:, t) = b;
            t = t + 1;
            
            %disp(W);
            %disp(b);
            
            % cada p esta escrito de forma traspuesta en el archivo
            % por lo que debemos aplicar la traspuesta
            p = P(j, :)';
            
            %fprintf(1, "p:\n");
            %disp(p);
            
            n = W * p + b;
            
            %fprintf(1, "n:\n");
            %disp(n);
            
            a = purelin(n);
            
            %fprintf(1, "target:\n");
            %disp(T(j, :)');
            %fprintf(1, "a:\n");
            %disp(a);
            
            % CALCULO DEL ERROR
            a_error(:, j) = T(j, :)' - a;
            
            %fprintf(1, "Error:\n");
            %disp(a_error(:, j));
            
            % CALCULO DE NUEVOS VALORES PARA W y b
            [W, b] = getNewValues(W, b, p, a_error(:, j), alfa);
        end
        EEPOCH = getEEPOCH(a_error);
        
        %fprintf(1, "EEPOCH:\n");
        %disp(EEPOCH);
        %fprintf(1, "\n\n");
        
        a_EEPOCH(:, i) = EEPOCH;
        
        
        % VERIFICANDO SI EEPOCH ES CERO
        [f_EEPOCH, c_EEPOCH] = size(EEPOCH);
        suma_EEPOCH = 0;
        for n = 1: c_EEPOCH
            suma_EEPOCH = suma_EEPOCH + ( EEPOCH(n, 1) == 0);
        end
        if suma_EEPOCH == c_EEPOCH
            fprintf(1, "Aprendizaje finalizado. EEPOCH = 0\n");
            break;
        else
            menores = 0;
            for n = 1: c_EEPOCH
                menores = menores + ( EEPOCH(n, 1) < Eepoch);
            end
            if menores == c_EEPOCH
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
    fprintf(1, "N° de epocas realizadas: %d\n", i);
    % GUARDAR VALORES DE W Y b
    dlmwrite('valores_finales_RNA.txt', W, 'delimiter', '\t');
    dlmwrite('valores_finales_RNA.txt', b, 'delimiter', '\t', '-append');
    
    fprintf(1, "EEPOCH = \n");
    disp(a_EEPOCH(:, i));
    
    fprintf(1, "W = \n");
    disp(W);
    
    fprintf(1, "b = \n");
    disp(b);
    
    hold off;
    plotW(a_W);
    plotb(a_b);
    plotEEPOCH(a_EEPOCH);
end

function W = getNewW(W_ant, p, e, alfa)
    W = W_ant + ( 2 * alfa )* e * p';
end

function initRegression(P, T, epochmax, Eepoch, alfa)
    [filas, columnas] = size(P);
    W = -1 + ( 1 + 1 )*rand( 1, columnas );
    % W = [0.84 0.39 0.78];
    % Arreglo de EEPOCH's y W's
    a_EEPOCH = [];
    a_W = [];
    % APRENDIZAJE
    for i = 1:epochmax
        % Arreglo de errores
        a_error = zeros( 1, filas );
        for j = 1:filas
            a_W(:, i) = W';
            %fprintf(1, "\t d%d\n", j);
            %disp(W);
            
            p = P(j,:)';
            n = W * p;
            a = purelin(n);
            
            % CALCULO DEL ERROR
            a_error(1,j) = ( T(j, 1) - a );
            %fprintf(1, "\tERROR=%.4f\n", a_error(1,j));
            
            % CALCULO DEL NUEVO VALOR PARA W
            W = getNewW(W, p, a_error(1, j), alfa);
        end
        EEPOCH = getEEPOCH(a_error);
        
        %fprintf(1, "\tEEPOCH=%.4f\n\n", EEPOCH);
        
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
    fprintf(1, "N° de epocas realizadas: %d\n", i);
    
    % GUARDAR VALORES DE W Y b
    dlmwrite('val_finales_RNA_reg.txt', W, 'delimiter', '\t');
    
    fprintf(1, "EEPOCH = \n");
    disp(a_EEPOCH(:, i));
    
    fprintf(1, "W = \n");
    disp(W);
    
    plotW(a_W);
    plotEEPOCH(a_EEPOCH);
end

function plotW(a_W)
    [filas, columnas] = size(a_W);
    
    figure(1);
    for i = 1:filas
        hold on;
        aux = a_W(i, :);
        plot(aux);
    end
    %plot(a_W);
    title("Evolucion de W");
end

function plotb(a_b)
    [filas, columnas] = size(a_b);
    
    figure(2);
    for i = 1:filas
        hold on;
        aux = a_b(i, :);
        plot( aux );
    end
    %plot(a_b);
    title("Evoluacion de b");
end

function plotEEPOCH(a_EEPOCH)
    [filas, columnas] = size(a_EEPOCH);
    
    figure(3);
    for i = 1:filas
        hold on;
        aux = a_EEPOCH(i,:);
        plot( aux );
    end    
    title("Evolucion de EEPOCH");
end



