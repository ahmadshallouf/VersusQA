package com.winlp.backend.dtos;

import java.util.List;

public record CamResponse(
        double firstObjectScore,
        List<Argument> firstObjectArguments,
        List<Argument> secondObjectArguments
) {

}
