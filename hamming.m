function hamming()

    fprintf(1, "------------------------------------- \n");
    fprintf(1, "\x7c           Red de Hamming          \x7c\n");
    fprintf(1, "------------------------------------- \n");
    
    W = dlmread( 'w1.txt' );
    p = dlmread( 'p_new.txt' );
    b = bias( W );
    a = feed_forward( W, p, b );
    me = epsilon( W );
    recurrencia( a, me );
    
end

function b = bias( W )
    [filas, columnas] = size(W);
    b = zeros( filas, 1 );
    for i = 1:filas
        b(i, 1) = columnas;
    end
end

function a = feed_forward( W, p, b )
    a = ( W * p ) + b;
end

function me = epsilon( W )
    [filas, columnas] = size( W );
    S = filas;
    lim = 1 / (S - 1);
    
    % e es un numero en el rango (0,lim)
    e = lim * rand( 1, 1 );
    fprintf(1, "Epsilon = %d\n", e);
    
    % me = matriz cuadrada
    me = ones( filas, filas );
    for i = 1:filas
        for j = 1:filas
            if i ~= j
                me( i, j ) = -e;
            end
        end
    end
end

function recurrencia( a, me )
    [filas, columnas] = size( a );
    % matriz que registra por columna el valor de a-esima
    an_total = a;
    iteracion = 1;
    comprobacion = 0;
    
    while iteracion < 150000
        fprintf(1, "Iteracion %d\n", iteracion);
        
        % clase del vector a clasificar
        clase = 0;
        
        % matriz a-enesima
        aux = zeros( filas, 1 );
        for z = 1:filas
           aux( z, 1 ) = an_total( z, iteracion );
        end
        an = me * aux;
        
        % bandera que almacenará la suma de los valores de a-enesima
        % si a-enesima[i]!=0 entonces sumaremos 1
        % si el valor final de la bandera es 1, significa que existe solo 1 
        % valor diferente de cero
        bandera_resp = 0;

        % poslin
        for i = 1:filas
            if an( i , 1) < 0
                an( i , 1) = 0;
            else
                clase = i;
            end
        end
        
        % almacenar valores obtenidos
        for j = 1: filas
            an_total( j, iteracion + 1 ) = an( j, 1 );
        end

        % calculo de bandera_resp
        for k = 1: filas
           if an( k, 1 ) ~= 0
               bandera_resp = bandera_resp + 1;
           end
        end
        
        % comprobar si existe solucion
        % si existe, hacemos una iteracion de comprobacion
        if bandera_resp == 1
            comprobacion = comprobacion + 1;
        else
            comprobacion = 0;
        end
        
        if comprobacion == 2
            % comprobar valores anteriores con actuales
            suma = 0;
            for m = 1:filas
                suma = suma + ( an( m, 1 ) - an_total( m, iteracion ) ); 
            end
            if suma == 0
                fprintf(1, "El vector pertenece a la clase %d\n", clase);
                disp( an_total );
                fprintf(1, "Fin\n");
                graficar( an_total );
                break;
            else
                comprobacion = 0;
            end
        end
        
        % incremento
        iteracion = iteracion + 1;
    end
end

function graficar( an_total )
    [filas, columnas] = size( an_total );
    M = zeros(columnas - 1, filas);
    
    for i = 1:filas
        for j = 1:columnas
            M( j, i ) = an_total( i, j );
        end
    end
    plot( M );
    xlabel('t(n)')
    ylabel('Clases')
end







