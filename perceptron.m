% 
% PERCEPTRON
% 
% Para usar este algoritmo se debe proporcionar un archivo con el nombre de
% perceptron_w.txt y que tenga la siguiente estructura
%
% W traspuesta
% targets traspuesta uno debajo de otro
% max_epoch
% 
function perceptron()
    fprintf(1, "------------------------------------- \n");
    fprintf(1, "\x7c             Perceptron             \x7c\n");
    fprintf(1, "------------------------------------- \n");
    init();
end

function init()
    cleanFolder();
    [P, T, max_epoch] = readValuesFile();
    procesar(P, T, max_epoch);
    % procesar nuevamente
    resp = input("¿Desea repetir el entrenamiento? SI/NO:", 's');
    if resp == "SI" || resp == "si"
        init();
    else
        fprintf(1, "ADIOS\n");
    end
end

function cleanFolder()
    hold off
    filename1 = "perceptron_w.txt";
    filename2 = "perceptron_b.txt";
    if isfile(filename1)
        delete perceptron_w.txt;
    end
    if isfile(filename2)
        delete perceptron_b.txt;
    end
end

function [P, T, max_epoch] = readValuesFile()
    valores = dlmread("perceptron_val.txt");
    [filas, columnas] = size(valores);
    j = 1;
    % nro_targets nos dirá cuantas clases debemos clasificar y a partir 
    % de esto obtendremos s
    nro_targets = 1;
    s = 0;
    t_aux = zeros((filas-1)/2, columnas);
    
    % P
    for i = 1:((filas - 1)/2)
        P(i, :) = valores(i,:);
    end
    
    % TARGETS
    for i = (filas+1)/2:(filas-1)
        t_aux(j) = valores(i);
        j = j + 1;
    end
    % asumiendo que los targets iguales estan uno tras otro
    for i = 2:(filas-1)/2
        if t_aux(i) ~= t_aux(i-1)
            nro_targets = nro_targets + 1;
        end
    end
    % calculamos s
    % s nos ayuda a saber la dimension de los targets sx1
    s = calcularS( nro_targets );
    for i = 1:( (filas-1)/2 )
        for k = 1:s
            T(i, k) = t_aux(i, k);
        end
    end
    
    % max_epoch
    max_epoch = valores(filas, 1);
end

function s = calcularS(nro_targets)
    % 2^s = nro_clases => s = log2(nro_clases)
    s = ceil(log2(nro_targets));
    fprintf(1, "S = %d\n\n", s);
end

function procesar(P, T, max_epoch)
    [filas, columnas] = size(P);
    % VALORES ALEATORIOS PARA W y b
    %
    % formula para nros aleatorios en un rango [a, b]
    % r = a + (b-a)*rand(N,1)
    W = -2 + (2+2)*rand(1, columnas);
    b = -5 + (5+5)*rand(1);
    
    % VARIABLES PARA CALCULAR EL ERROR
    %
    % e es una matriz para calcular los errores de cada iteracion durante 1
    % epoca
    e = zeros(1, filas);
    % si la suma de errores durante una epoca es cero entonces los
    % elementos estan clasificados
    sum_e = 0;
    
    for i = 1:max_epoch
        fprintf(1, "EPOCA %d\n\n", i);
        for j = 1:filas
            fprintf(1, "iteracion %d\n\n", j);
            disp(W);
            disp(b);
            saveW(W);
            saveB(b);
            n = 0;
            a = 0;
            % producto de W y P
            for k = 1:columnas
                n = n + W(1,k)*P(j, k);
            end
            % sumamos el bias
            n = n + b;
            %%fprintf(1, "\tn = %d\n\n", n);
            
            % HARDLIM
            if n >= 0
                a = 1;
            end
            
            % calculo del error
            e(1, j) = T(j, 1) - a;
            
            % nuevos valores para W y b
            [W, b] = setNewValues(W, b, P, e(1, j), j);
            fprintf(1, "\tError = %d\n\n", e(1, j));
        end
        % Verificar que el error de cada iteracion es 0
        sum_e = 0;
        for z = 1: filas
            sum_e = sum_e + (e(1,z) == 0);
        end
        if sum_e == filas
            break;
        end
    end
    if i ~= max_epoch
        fprintf(1, "Aprendizaje culminado: datos del conjunto de entrenamiento clasificados\n\n");
    else
        if i == max_epoch && sum_e == filas
            fprintf(1, "Aprendizaje culminado: datos del conjunto de entrenamiento clasificados\n\n");
        else
            fprintf(1, "Aprendizaje culminado: max_epoch alcanzado\n\n");
        end
    end
    plotValues();
end

function [W, b] = setNewValues(W, b, P, e, dato)
    [filas, columnas] = size(W);
    for i = 1:columnas
        W(1, i) = W(1, i) + e*P(dato, i);
    end
    b = b + e;
end

function saveW(W)
    filename = "perceptron_w.txt";
    if isfile(filename)
        dlmwrite(filename, W, '-append', 'delimiter', '\t', 'roffset', 1);
    else
        dlmwrite(filename, W, 'delimiter', '\t', 'roffset', 1);
    end
end

function saveB(b)
    filename = "perceptron_b.txt";
    if isfile(filename)
        dlmwrite(filename, b, '-append', 'delimiter', '\t', 'roffset', 1);
    else
        dlmwrite(filename, b, 'delimiter', '\t', 'roffset', 1);
    end
end

function plotValues()
    W = dlmread("perceptron_w.txt");
    b = dlmread("perceptron_b.txt");
    [x_max, y_min, y_max] = findLimits(); 
    plot( W );
    hold on
    plot( b, 's-m','MarkerSize', 6 );
    % Rango para mayor apreciacion de resultados
    axis([0 (x_max+0.5) (y_min-0.5) (y_max+0.5)]);
    fclose('all');
end

function [x_max, y_min, y_max] = findLimits()
    W = dlmread("perceptron_w.txt");
    b = dlmread("perceptron_b.txt");
    [filas, columnas] = size(W);
    x_max = filas;
    y_min = 99;
    y_max = -1;
    for i = 1:filas
        for j = 1:columnas
            if y_max < W(i,j)
                y_max = W(i,j);
            end
            if y_min > W(i,j)
                y_min = W(i,j);
            end
        end
    end
    for k = 1:filas
        if y_max < b(k,1)
            y_max = b(k,1);
        end
        if y_min > b(k,1)
            y_min = b(k,1);
        end
    end
end
