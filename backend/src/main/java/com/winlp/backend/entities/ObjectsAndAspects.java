package com.winlp.backend.entities;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Entity
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ObjectsAndAspects {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE)
    private Long id;

    private String question;

    private String object1;
    private String object2;

    private String aspect1;
    private String aspect2;
    private String aspect3;
    private String aspect4;
    private String aspect5;

    public ObjectsAndAspects(String question, String object1, String object2, String aspect1, String aspect2, String aspect3, String aspect4, String aspect5) {
        this.question = question;
        this.object1 = object1;
        this.object2 = object2;
        this.aspect1 = aspect1;
        this.aspect2 = aspect2;
        this.aspect3 = aspect3;
        this.aspect4 = aspect4;
        this.aspect5 = aspect5;
    }
}
