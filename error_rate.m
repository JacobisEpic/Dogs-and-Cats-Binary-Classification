function error_rate_value = error_rate(yguess,ytrue)

    if (~(iscolumn(ytrue)))
        error("ytrue is not a column vector.")
    elseif (~(iscolumn(yguess)))
        error("yguess is not a column vector.")
    elseif (length(ytrue)~=length(yguess))
        error("ytrue and yguess are not the same length.")
    end

    error_rate_value = 1/length(yguess)*sum(yguess ~= ytrue);