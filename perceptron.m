function perceptron()
    fprintf(1, "------------------------------------- \n");
    fprintf(1, "\x7c             Perceptron             \x7c\n");
    fprintf(1, "------------------------------------- \n");
    [P, T, max_epoch] = leerArchivo();
    procesar(P, T, max_epoch);
end

function [P, T, max_epoch] = leerArchivo()
    valores = dlmread("val_perceptron.txt");
    [filas, columnas] = size(valores);
    j = 1;
    for i = 1:((filas - 1)/2)
        P(i, :) = valores(i,:);
    end
    for i = (filas+1)/2:(filas-1)
        T(j, 1) = valores(i, 1);
        j = j + 1;
    end
    max_epoch = valores(filas, 1);
end

function [W, b] = newValues(W, b, P, e, dato)
    [filas, columnas] = size(W);
    for i = 1:columnas
        W(1, i) = W(1, i) + e*P(dato, i);
    end
    b = b + e;
end

function procesar(P, T, max_epoch)
    [filas, columnas] = size(P);
    % Valores aleatorios para W y b
    % formula para nros aleatorios en un rango [a, b]
    % r = a + (b-a)*rand(N,1)
    W = -2 + (2+2)*rand(1, columnas);
    b = -5 + (5+5)*rand(1);
    e = zeros(1, filas);
    sum_e = 0;
    for i = 1:max_epoch
        fprintf(1, "EPOCA %d\n\n", i);
        for j = 1:filas
            fprintf(1, "iteracion %d\n\n", j);
            disp(W);
            disp(b);
            n = 0;
            a = 0;
            % producto de W y P
            for k = 1:columnas
                n = n + W(1,k)*P(j, k);
            end
            % sumamos el bias
            n = n + b;
            fprintf(1, "\tn = %d\n\n", n);
            % hardlim
            if n >= 0
                a = 1;
            end
            % calculo del error
            e(1, j) = T(j, 1) - a;
            % nuevos valores para W y b
            [W, b] = newValues(W, b, P, e(1, j), j);
            %fprintf(1, "\tError %d = %d\n\n", j, e(1, j));
            fprintf(1, "\tError = %d\n\n", e(1, j));
        end
        % Verificar que el error de cada iteracion es 0
        sum_e = 0;
        for z = 1: filas
            sum_e = sum_e + (e(1,z) == 0);
        end
        % fprintf(1, "Error total = %d\n\n", sum_e);
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
end

















