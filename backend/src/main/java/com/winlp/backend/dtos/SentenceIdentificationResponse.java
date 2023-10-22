package com.winlp.backend.dtos;

import java.util.List;

/**
 class response(BaseModel):
 arguments1: list[argument]
 arguments2: list[argument]

 */
public record SentenceIdentificationResponse(
        List<Argument> arguments1,
        List<Argument> arguments2
) {
}
