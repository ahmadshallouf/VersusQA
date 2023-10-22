package com.winlp.backend.dtos;

import java.util.List;

public record SummaryRequest(
        String object1,
        String object2,
        List<String> arguments
)
{

}
