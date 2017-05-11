function pflag = add_pt(ng, rm, rc, pp, po_m, pt_m, ip)

% To add a point if other neighborhood points far enough away

% Neighborhood is (2*rc+1)^2 cells, where rc is an integer
% Possible point postition is pp (x + i*y) (k-space)
pflag = 1 ;

% Get cell (m,n) of possible point
px = real(pp) ;     py = imag(pp) ;
m = ng/2 + 1 + round(py) ;    n = ng/2 + 1 + round(px) ;

% Ensure that possible point is not in same cell as initiating point
if rm < sqrt(2)
    if (round(imag(ip)) == m && round(real(ip)) == n)
        pflag = 0 ;
        return
    end
end

% See if proposed point is within FOV
if (m < 1 || m > ng) || (n < 1 || n > ng)
    pflag = 0 ;
    return
end

% Look over neighborhood for points, and calculate radius
for r = (m-rc):(m+rc)
    for s = (n-rc):(n+rc)
        % Make sure cell is within FOV
        if (r > 0 && r < ng+1 && s > 0 && s < ng+1)  
            if po_m(r,s) == 1
                pxn = real(pt_m(r,s));  pyn = imag(pt_m(r,s)) ;
                rad = sqrt((pxn-px)^2 + (pyn-py)^2) ;
                % If too close, reject
                if rad < rm 
                    pflag = 0 ;
                    return
                end
            end
        end
    end
end
        
% END OF FUNCTION
        