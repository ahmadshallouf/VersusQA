import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ArgumentsPresentationComponent } from './arguments-presentation.component';

describe('ArgumentsPresentationComponent', () => {
  let component: ArgumentsPresentationComponent;
  let fixture: ComponentFixture<ArgumentsPresentationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ArgumentsPresentationComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ArgumentsPresentationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
