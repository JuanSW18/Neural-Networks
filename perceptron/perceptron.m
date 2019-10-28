% 
% PERCEPTRON
% 
% Para usar este algoritmo se debe proporcionar un archivo con el nombre de
% perceptron_w.txt y que tenga la siguiente estructura
%
% p's traspuesta
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
    [S, P, T, max_epoch] = readValuesFromFile();
    startLearning(S, P, T, max_epoch);
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
        delete perceptron_w*.txt;
    end
    if isfile(filename2)
        delete perceptron_b*.txt;
    end
end

function [S, P, T, max_epoch] = readValuesFromFile()
    valores = dlmread("perceptron_val.txt");
    [filas, columnas] = size(valores);
    j = 1;
    % nro_targets nos dirá cuantas clases debemos clasificar y a partir 
    % de esto obtendremos s
    nro_targets = 1;
    S = 0;
    t_aux = zeros((filas-1)/2, columnas);
    
    % P
    for i = 1:((filas - 1)/2)
        P(i, :) = valores(i,:);
    end
    
    % TARGETS
    for i = (filas+1)/2:(filas-1)
        t_aux(j, :) = valores(i, :);
        j = j + 1;
    end
    % asumiendo que los targets iguales estan uno tras otro
    for i = 2:(filas-1)/2
        f1_aux = t_aux(i, :);
        f2_aux = t_aux(i-1, :);
        if ~isequal(f1_aux, f2_aux)
            nro_targets = nro_targets + 1;
        end
    end
    % calculamos s
    % s nos ayuda a saber la dimension de los targets sx1
    S = calculateS( nro_targets );
    for i = 1:( (filas-1)/2 )
        for k = 1:S
            T(i, k) = t_aux(i, k);
        end
    end
    
    % max_epoch
    max_epoch = valores(filas, 1);
end

function s = calculateS(nro_targets)
    % 2^s = nro_clases => s = log2(nro_clases)
    fprintf(1, "N° de targets = %d\n", nro_targets);
    s = ceil(log2(nro_targets));
    fprintf(1, "S = %d\n", s);
end

function startLearning(S, P, T, max_epoch)
    % n = W*p + b y a = hardlim(W*p + b)
    % W tiene una dimension de SxR
    % p tiene una dimension de Rx1
    % b tiene una dimension de Sx1
    % n tiene una dimension de Sx1
    % a tiene una dimension de Sx1
    
    % R es igual al numero de filas
    [filas, columnas] = size( P );
    % VALORES ALEATORIOS PARA W y b
    %
    % formula para nros aleatorios en un rango [a, b]
    % r = a + (b-a)*rand(N,1)
    W = -2 + ( 2 + 2 )*rand( S, columnas );
    b = -5 + ( 5 + 5 )*rand( S, 1 );
    
    % valores para ejemplo 2: ok
    % W = [-2 -1; 3 -4];
    % b = [1 ; 2];
    
    % valores para ejemplo 3: por revisar
    % debemos eliminar la columna de ceros de mas
    %W = [1 1; -1 1; -1 -1];
    %b = [-6 ; -2; -3];
    
    % VARIABLES PARA CALCULAR EL ERROR
    %
    % e es una matriz para calcular los errores de cada iteracion durante 1
    % epoca
    e = zeros( S, filas );
    % si la suma de errores durante una epoca es cero entonces los
    % elementos estan clasificados
    sum_e = 0;
    
    for i = 1:max_epoch
        fprintf(1, "EPOCA %d\n", i);
        for j = 1:filas
            fprintf(1, "iteracion %d\n", j);
%           disp( "   W     " );
            %disp( W );
%           disp( "   b     " );
            %disp( b );
            
            % guardamos los valores de W y b
            saveW( W );
            saveB( b );
            saveMultiW(W, S);
            saveMultiB(b, S);
            n = zeros( S, 1 );
            a = zeros( S, 1 );
            
            % producto de W y P
            for k = 1:columnas
                for m = 1:S
                    n(m, 1) = n(m, 1) + W(m, k)*P(j, k);
                end
            end

            % sumamos el bias
            n = n + b;
            
            % HARDLIM
            for k = 1:S
                if n(k, 1) >= 0
                    a(k, 1) = 1;
                end
            end
            
%           disp( "   a     " );
%           disp(a);
            
            % calculo del error
            for k = 1:S
                e(k, j) = T(j, k) - a(k, 1);
            end
            
            % nuevos valores para W y b
            [W, b] = setNewValues(W, b, P, e(:, j), j);
            
            fprintf(1, "\tError = ");
            [f, c] = size(e);
            for k = 1:f
                fprintf(1, "%d ", e(k,j));
            end
            fprintf(1, "\n");
        end
        fprintf(1, "\n");
        % Verificar que el error de cada iteracion es 0
        sum_e = 0;
        for k = 1: filas
            for m = 1:S
                sum_e = sum_e + (e(m,k) == 0);
            end
        end
        if ( sum_e/S ) == filas
            break;
        end
    end
    if i ~= max_epoch
        fprintf(1, "Aprendizaje culminado: datos del conjunto de entrenamiento clasificados\n\n");
    else
        if i == max_epoch && ( sum_e/S ) == filas
            fprintf(1, "Aprendizaje culminado: datos del conjunto de entrenamiento clasificados\n\n");
        else
            fprintf(1, "Aprendizaje no culminado: max_epoch alcanzado\n\n");
        end
    end
    plotMultiValues(S)
end

function [W, b] = setNewValues(W, b, P, e, dato)
    W = W + e*P(dato, :);
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

function saveMultiW(W, S)
    for i = 1:S
        filename = "perceptron_w_c" + i + ".txt";
        if isfile(filename)
            dlmwrite(filename, W(i,:), '-append', 'delimiter', '\t', 'roffset', 1);
        else
            dlmwrite(filename, W(i,:), 'delimiter', '\t', 'roffset', 1);
        end
    end
end

function saveMultiB(b, S)
    for i = 1:S
        filename = "perceptron_b_c" + i + ".txt";
        if isfile(filename)
            dlmwrite(filename, b(i,:), '-append', 'delimiter', '\t', 'roffset', 1);
        else
            dlmwrite(filename, b(i,:), 'delimiter', '\t', 'roffset', 1);
        end
    end
end

function plotMultiValues(S)
    Y = plotFrontier(S);
    if S == 1
        W = dlmread("perceptron_w.txt");
        b = dlmread("perceptron_b.txt");
        [x_max, y_min, y_max] = findLimitsInAllFiles(S);
        plot( W );
        hold on
        plot( b, 's-m', 'DisplayName','bias' );
        grid on;
        legend;
        % Rango para mayor apreciacion de resultados
        axis([0 (x_max+2) (y_min-0.5) (y_max+0.5)]);
    else
        marker = ['o' '+' '*' 'd' 'p' 'h'];
        color = ['r' 'g' 'b' 'm' 'c'];
        [x_max, y_min, y_max] = findLimitsInAllFiles(S);
        for i = 1:S
            filename = "perceptron_w_c" + i + ".txt";
            W = dlmread(filename);
            w_name = "w" + i;
            plot( W, 'LineStyle', '-', 'Color', color(i), 'Marker', marker(i), 'DisplayName', w_name);
            hold on
        end
        for i = 1:S
            filename = "perceptron_b_c" + i + ".txt";
            b = dlmread(filename);
            plot( b, 's-m', 'DisplayName','bias' );
            hold on
        end
        grid on;
        legend;
        % Rango para mayor apreciacion de resultados
        axis([0 (x_max+2) (y_min-0.5) (y_max+0.5)]);
    end
    plot( Y );
    fclose('all');
end

function [x_max, y_min, y_max] = findLimitsInAllFiles(S)
    x_max = -99;
    y_min = 99;
    y_max = -1;
    for r = 1:S
        w_filename = "perceptron_w_c" + r + ".txt";
        b_filename = "perceptron_b_c" + r + ".txt";
        W = dlmread(w_filename);
        b = dlmread(b_filename);
        [filas, columnas] = size(W);
        x_max = filas;
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
end

function Y = plotFrontier(S)
    W = dlmread("perceptron_w.txt");
    b = dlmread("perceptron_b.txt");
    [filas, columnas] = size(W);
    m1 = (-1)*( W(filas, 2)/ W(filas, 1) );
    m2 = (-1)/m1;
    if S == 1
       last_w = W(filas, :);
       last_b = b(filas, :);
       p = (-1)*(last_w\last_b);
       b = m2*p(2,1) - p(1,1);
       x = -2:6;
       Y = m2*x + b;
    else
        
    end
end
