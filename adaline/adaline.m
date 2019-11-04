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
    modo = input("Ingrese R(Regresor) o C(Clasificador) segun el tipo de problema: ", 's');
    if modo == "r" || modo == "R"
        fprintf(1, "Empezando regresion ...\n");
    else
        fprintf(1, "Empezando clasificacion ...\n");
    end
    epochmax = input("Ingrese el valor de epochmax: ");
    e_epoch = input("Ingrese el valor de eepoch: ");
end

% Leer archivos para el dataset
function [P, T] = readValuesFromFiles()
    P = dlmread("p.txt");
    T = dlmread("targets.txt");
end

% Calcular el numero de clases
function classes_number = getNumberOfClasses(T)
    [filas, columnas] = size(T);
    classes_number = 0;
    % asumiendo que los targets iguales estan uno tras otro
    for i = 2:filas
        f1_aux = T(i, :);
        f2_aux = T(i-1, :);
        if ~isequal(f1_aux, f2_aux)
            nro_targets = nro_targets + 1;
        end
    end
end

% Imprimir el modelo matematico
% a = purelin(W*p + b)
% S*1      S*R R*1 S*1 
function printMathematicModel()
    fprintf(1, "...\n");
end

function eepoch = getEEPOCH(e)
    [filas, columnas] = size(e);
    suma = 0;
    for i = 1: columnas
        suma = suma + e(1, i);
    end
    eepoch = suma / columnas;
end



