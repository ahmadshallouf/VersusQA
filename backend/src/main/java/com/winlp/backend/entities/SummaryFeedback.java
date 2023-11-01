package com.winlp.backend.entities;

import jakarta.persistence.*;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Entity
@Data
@AllArgsConstructor
@NoArgsConstructor
public class SummaryFeedback {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE)
    private Long id;

    private String question;

    // long string
    @Column(columnDefinition = "TEXT")
    private String arguments;

    // long string
    @Column(columnDefinition = "TEXT")
    private String summary;

    private boolean useful;
    private boolean fluent;

    public SummaryFeedback(String question, String arguments, String summary, boolean useful, boolean fluent) {
        this.question = question;
        this.arguments = arguments;
        this.summary = summary;
        this.useful = useful;
        this.fluent = fluent;
    }

}
