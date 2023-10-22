import { ComponentFixture, TestBed } from '@angular/core/testing';

import { QuestionAskingComponent } from './question-asking.component';

describe('QuestionAskingComponent', () => {
  let component: QuestionAskingComponent;
  let fixture: ComponentFixture<QuestionAskingComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ QuestionAskingComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(QuestionAskingComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
