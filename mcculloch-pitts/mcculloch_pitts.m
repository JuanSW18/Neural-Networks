function mcculloch_pitts()
    
    fprintf(1, " ------------------------------------- \n");
    fprintf(1, "\x7c      Célula de McCulloch-Pitts      \x7c\n");
    fprintf(1, "\x7c Elija una compuerta:                \x7c\n");
    fprintf(1, "\x7c 1.NOT                               \x7c\n");
    fprintf(1, "\x7c 2.AND                               \x7c\n");
    fprintf(1, "\x7c 3.OR                                \x7c\n");
    fprintf(1, " ------------------------------------- \n");
    compuerta = input("N° de compuerta: ");
    if compuerta == 2 || compuerta == 3
        n = input("Escriba la dimension de la compuerta: ");
        dataset = tabla_de_verdad(compuerta, n);
        procesar(dataset);
    else
        dataset = tabla_de_verdad(compuerta, 0);
        procesar(dataset);
    end
    
end

function Y = tabla_de_verdad(compuerta, n)

    if n == 0
        Y = [0 1; 1 0];
    else
        tam = 2^n;
        Y = zeros(tam, n);
        for i = 2:tam
            for j = 1:n
                Y(i,j) = Y(i-1,j);
            end
            Y(i,n) = Y(i,n) + 1;
            for k = n:-1:2
                if Y(i,k) == 2
                    Y(i,k) = 0;
                    Y(i,k-1) = Y(i, k-1) + 1;
                end
            end
        end
        if compuerta == 2
            for i = 1:tam
                Y(i, n+1) = 0;
            end
            Y(tam, n+1) = 1;
        else
            for i = 1:tam
                Y(i, n+1) = 1;
            end
            Y(1, n+1) = 0;
        end
    end

end

function procesar(dataset)
    [filas, columnas] = size(dataset);
    teta = randi([-2^23,2^23]);
    W = randi([-2^23,2^23], 1, (columnas-1));
    a = -1;
    total = 1;
    epoch = input("Ingrese nro de epocas: ");
    for e = 1:epoch
        a = -1;
        n = 0;
        respuesta = [];
        total = 1;
        for i = 1:filas
            for j = 1:(columnas-1)
                n = n + (W(1, j) * dataset(i, j));
            end

            if n > teta
                a = 1;
            else
                a = 0;
            end
            respuesta(1,i) = (a == dataset(i,columnas));
            if a ~= dataset(i,columnas)
                W = randi([-2^23,2^23], 1, (columnas-1));
            end
        end
        for p = 1:(filas)
            total = total * respuesta(1, p);
        end
        if total == 1
            fprintf("EXITO\n");
            fileID = fopen('val_finales.txt', 'at');
            dlmwrite('val_finales.txt', W, 'delimiter', '\t');
            fprintf(fileID, '\n\nteta=%d', teta);
            fclose('all');
            fprintf("\n");
            break;
        end
        
    end
    
    if total == 0
        fprintf("APRENDIZAJE FALLIDO\n");
        continuar = input("¿Desea continua s(si) o n(no)?", 's');
        if strcmp(continuar, 's')
            procesar(dataset);
        end
    end

end








