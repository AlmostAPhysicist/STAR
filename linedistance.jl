using LinearAlgebra
function distance(line1::Tuple{Vector}, line2::Tuple{Vector})
    d1 = line[1]-line[2]
    n = cross(line1[2], line2[2])

end