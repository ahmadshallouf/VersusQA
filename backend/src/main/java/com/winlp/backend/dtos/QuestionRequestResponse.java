package com.winlp.backend.dtos;
import java.util.List;

public record QuestionRequestResponse(String question,
                                      boolean isComparative,
                                      List<String> objects,
                                      List<String> aspects,
                                      String answer) {
}
