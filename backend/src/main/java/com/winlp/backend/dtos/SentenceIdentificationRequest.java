package com.winlp.backend.dtos;

import java.util.List;

/**
 * class request(BaseModel):
 *     object1: str
 *     object2: str
 *     arguments: list[argument]
 */
public record SentenceIdentificationRequest(
        String object1,
        String object2,
        List<Argument> arguments
) {
}
