package com.winlp.backend.dtos;

public record SummaryFeedbackRequest(
        String question,
        String arguments,
        String summary,
        boolean useful,
        boolean fluent
) {
}
